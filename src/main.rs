mod gguf; mod tensor; mod model; mod math; mod tokenizer; mod sampler; mod gpu;

use std::io::{self,BufRead,Write};
use std::path::Path;
use anyhow::Result;
use clap::Parser;
use model::llama::{KvCache,LlamaModel};
use tokenizer::bpe::Tokenizer;
use tokenizer::chat::ChatTemplate;
use gpu::GpuCtx;

#[derive(Parser)]
#[command(name="gguf-cli")]
struct Args {
    #[arg(short,long)] model: String,
    #[arg(short,long)] prompt: Option<String>,
    #[arg(short,long,default_value="You are a helpful assistant.")] system: String,
    #[arg(short='n',long,default_value_t=512)] max_tokens: usize,
    #[arg(short='t',long,default_value_t=0.8)] temperature: f32,
    #[arg(short='c',long,default_value_t=8192)] ctx_len: usize,
    /// Use GPU acceleration (requires NVIDIA/AMD GPU with Vulkan or DX12)
    #[arg(long,default_value_t=false)] gpu: bool,
    #[arg(long,default_value_t=42)] seed: u64,
}

fn main()->Result<()>{
    let args=Args::parse();
    sampler::set_seed(args.seed);

    let path=Path::new(&args.model);
    let (model,gguf)=LlamaModel::load(path)?;
    let tok=Tokenizer::from_gguf(&gguf)?;

    let c=&model.config;
    let ctx_len=args.ctx_len.min(c.n_ctx);
    eprintln!("Context: {} tokens", ctx_len);

    let gpu: Option<GpuCtx> = if args.gpu {
        match GpuCtx::init() {
            Some(g) => { eprintln!("GPU mode enabled."); Some(g) }
            None    => { eprintln!("GPU init failed — falling back to CPU."); None }
        }
    } else { None };

    let mut cache=KvCache::new(c.n_layers,ctx_len,c.n_kv_heads,c.head_dim());

    match args.prompt {
        Some(ref p) => {
            print!("{}",p); io::stdout().flush()?;
            let ids=tok.encode(p,true);
            let (mut pos,mut logits)=prefill(&model,&mut cache,&ids,0,gpu.as_ref());
            generate(&model,&tok,&mut cache,&mut pos,&mut logits,args.max_tokens,args.temperature,ctx_len,&[tok.eos_id],gpu.as_ref());
            println!();
        }
        None => {
            let tmpl=ChatTemplate::detect(&tok);
            let stops=tmpl.stop_tokens(&tok);
            eprintln!("Template: {:?} | /quit to exit\n",tmpl);
            let mut pos=0usize;

            let sys=tmpl.system_prompt(&args.system);
            if !sys.is_empty(){
                let ids=tok.encode(&sys,true);
                let(p,_)=prefill(&model,&mut cache,&ids,pos,gpu.as_ref());
                pos=p;
            }

            let stdin=io::stdin();
            loop{
                eprint!("\nYou: "); io::stderr().flush()?;
                let mut line=String::new();
                if stdin.lock().read_line(&mut line)?==0{break;}
                let msg=line.trim();
                if msg.is_empty(){continue;}
                if msg=="/quit"||msg=="/exit"{break;}

                let turn=tmpl.user_turn(msg);
                let ids=tok.encode(&turn,pos==0);
                let(new_pos,mut logits)=prefill(&model,&mut cache,&ids,pos,gpu.as_ref());
                pos=new_pos;

                eprint!("Assistant: "); io::stderr().flush()?;
                generate(&model,&tok,&mut cache,&mut pos,&mut logits,args.max_tokens,args.temperature,ctx_len,&stops,gpu.as_ref());
                println!();

                if pos>=ctx_len.saturating_sub(64){
                    eprintln!("[Context full — resetting]");
                    cache=KvCache::new(c.n_layers,ctx_len,c.n_kv_heads,c.head_dim());
                    pos=0;
                }
            }
        }
    }
    Ok(())
}

fn prefill(model:&LlamaModel,cache:&mut KvCache,ids:&[u32],start:usize,gpu:Option<&GpuCtx>)->(usize,Vec<f32>){
    let mut logits=vec![0f32;model.config.n_vocab];
    for(i,&id) in ids.iter().enumerate(){logits=model.forward(id as usize,start+i,cache,gpu);}
    (start+ids.len(),logits)
}

fn generate(model:&LlamaModel,tok:&Tokenizer,cache:&mut KvCache,pos:&mut usize,last:&mut Vec<f32>,max:usize,temp:f32,ctx:usize,stops:&[u32],gpu:Option<&GpuCtx>)->usize{
    let mut n=0;
    loop{
        let next=sampler::sample(last,temp);
        if stops.contains(&(next as u32)){break;}
        if *pos>=ctx-1||n>=max{break;}
        print!("{}",tok.decode(next as u32)); io::stdout().flush().ok();
        *last=model.forward(next,*pos,cache,gpu);
        *pos+=1; n+=1;
    }
    n
}