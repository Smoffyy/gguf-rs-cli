use std::collections::HashMap;
use crate::gguf::types::GgufFile;

#[derive(Debug,PartialEq)] pub enum TokModel { Llama, Gpt2 }

pub struct Tokenizer {
    pub vocab:       Vec<String>,
    pub scores:      Vec<f32>,
    pub token_to_id: HashMap<String,u32>,
    merge_rank:      HashMap<(String,String),usize>,
    byte_enc:        [char;256],
    byte_dec:        HashMap<char,u8>,
    special:         Vec<String>,
    tok_model:       TokModel,
    pub bos_id:      u32,
    pub eos_id:      u32,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> anyhow::Result<Self> {
        let tokens = gguf.metadata.get("tokenizer.ggml.tokens")
            .and_then(|v|v.as_arr())
            .ok_or_else(||anyhow::anyhow!("Missing tokenizer.ggml.tokens"))?;
        let vocab:Vec<String>=tokens.iter().map(|v|v.as_str().unwrap_or("").to_string()).collect();
        let scores:Vec<f32>=gguf.metadata.get("tokenizer.ggml.scores")
            .and_then(|v|v.as_arr())
            .map(|a|a.iter().map(|v|v.as_f32().unwrap_or(0.0)).collect())
            .unwrap_or_else(||vec![0.0;vocab.len()]);
        let token_to_id:HashMap<String,u32>=vocab.iter().enumerate().map(|(i,s)|(s.clone(),i as u32)).collect();
        let bos_id=gguf.metadata.get("tokenizer.ggml.bos_token_id").and_then(|v|v.as_u32()).unwrap_or(1);
        let eos_id=gguf.metadata.get("tokenizer.ggml.eos_token_id").and_then(|v|v.as_u32()).unwrap_or(2);
        let tok_model=match gguf.metadata.get("tokenizer.ggml.model").and_then(|v|v.as_str()){
            Some("gpt2")=>TokModel::Gpt2, _=>TokModel::Llama
        };
        let byte_enc=build_byte_encoder();
        let byte_dec:HashMap<char,u8>=byte_enc.iter().enumerate().map(|(b,&c)|(c,b as u8)).collect();
        let mut merge_rank=HashMap::new();
        if let Some(arr)=gguf.metadata.get("tokenizer.ggml.merges").and_then(|v|v.as_arr()){
            for (rank,v) in arr.iter().enumerate(){
                if let Some(s)=v.as_str(){ if let Some(sp)=s.find(' '){
                    let a=s[..sp].to_string(); let b=s[sp+1..].to_string();
                    merge_rank.insert((a,b),rank);
                }}
            }
        }
        let token_types:Vec<u32>=gguf.metadata.get("tokenizer.ggml.token_type")
            .and_then(|v|v.as_arr())
            .map(|a|a.iter().map(|v|v.as_u32().unwrap_or(0)).collect())
            .unwrap_or_default();
        let mut special:Vec<String>=vocab.iter().enumerate()
            .filter(|(i,s)|{
                let t=token_types.get(*i).copied().unwrap_or(0);
                t==2||t==3||(s.starts_with("<|")&&s.ends_with("|>"))||(s.starts_with('[')&&s.ends_with(']')&&s.len()>2)
            }).map(|(_,s)|s.clone()).collect();
        special.sort_by(|a,b|b.len().cmp(&a.len())); special.dedup();
        Ok(Self{vocab,scores,token_to_id,merge_rank,byte_enc,byte_dec,special,tok_model,bos_id,eos_id})
    }

    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut ids=if add_bos{vec![self.bos_id]}else{vec![]};
        for (seg,is_sp) in self.split_special(text){
            if is_sp { if let Some(&id)=self.token_to_id.get(&seg){ids.push(id);} }
            else { self.encode_seg(&seg,&mut ids); }
        }
        ids
    }

    fn encode_seg(&self, text: &str, ids: &mut Vec<u32>){
        if text.is_empty(){return;}
        let mut syms:Vec<String>=match self.tok_model{
            TokModel::Llama=>format!(" {}",text).chars()
                .map(|c|if c==' '{"\u{2581}".to_string()}else{c.to_string()}).collect(),
            TokModel::Gpt2=>text.bytes().map(|b|self.byte_enc[b as usize].to_string()).collect(),
        };
        match self.tok_model{TokModel::Llama=>self.bpe_score(&mut syms),TokModel::Gpt2=>self.bpe_rank(&mut syms)}
        for sym in &syms{
            if let Some(&id)=self.token_to_id.get(sym){ids.push(id);}
            else{for byte in sym.as_bytes(){let k=format!("<0x{:02X}>",byte);if let Some(&id)=self.token_to_id.get(&k){ids.push(id);}}}
        }
    }

    fn bpe_score(&self, s: &mut Vec<String>){
        loop{
            let mut best=f32::NEG_INFINITY; let mut bi=None;
            for i in 0..s.len().saturating_sub(1){
                let m=format!("{}{}",s[i],s[i+1]);
                if let Some(&id)=self.token_to_id.get(&m){let sc=self.scores[id as usize];if sc>best{best=sc;bi=Some(i);}}
            }
            match bi{None=>break,Some(i)=>{let m=format!("{}{}",s[i],s[i+1]);s[i]=m;s.remove(i+1);}}
        }
    }

    fn bpe_rank(&self, s: &mut Vec<String>){
        loop{
            let mut best=usize::MAX; let mut bi=None;
            for i in 0..s.len().saturating_sub(1){
                let p=(s[i].clone(),s[i+1].clone());
                if let Some(&r)=self.merge_rank.get(&p){if r<best{best=r;bi=Some(i);}}
            }
            match bi{None=>break,Some(i)=>{let m=format!("{}{}",s[i],s[i+1]);s[i]=m;s.remove(i+1);}}
        }
    }

    fn split_special(&self, text: &str) -> Vec<(String,bool)>{
        let mut res=vec![]; let mut cur=String::new();
        let chars:Vec<char>=text.chars().collect(); let mut i=0;
        'o: while i<chars.len(){
            let rest:String=chars[i..].iter().collect();
            for sp in &self.special{if rest.starts_with(sp.as_str()){
                if !cur.is_empty(){res.push((std::mem::take(&mut cur),false));}
                res.push((sp.clone(),true)); i+=sp.chars().count(); continue 'o;
            }}
            cur.push(chars[i]); i+=1;
        }
        if !cur.is_empty(){res.push((cur,false));} res
    }

    pub fn decode(&self, id: u32) -> String {
        if id as usize>=self.vocab.len(){return String::new();}
        let s=&self.vocab[id as usize];
        match self.tok_model{
            TokModel::Llama=>s.replace('\u{2581}'," "),
            TokModel::Gpt2=>{
                let bytes:Vec<u8>=s.chars().filter_map(|c|self.byte_dec.get(&c).copied()).collect();
                if bytes.len()==s.chars().count(){String::from_utf8_lossy(&bytes).into_owned()}
                else{s.clone()}
            }
        }
    }
}

fn build_byte_encoder()->[char;256]{
    let mut enc=['\0';256]; let mut mapped=[false;256];
    for &(lo,hi) in &[(33usize,126),(161,172),(174,255)]{
        for b in lo..=hi{enc[b]=char::from_u32(b as u32).unwrap();mapped[b]=true;}
    }
    let mut n=0u32;
    for b in 0usize..256{if !mapped[b]{enc[b]=char::from_u32(256+n).unwrap_or('?');n+=1;}}
    enc
}