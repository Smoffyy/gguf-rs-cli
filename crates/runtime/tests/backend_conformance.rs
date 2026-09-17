//! Every backend must compute the same thing.
//!
//! The CPU backend is the reference: it is the simplest code and the one that is easiest to
//! read against the format documentation. Each GPU backend is checked op by op against it
//! with identical inputs.
//!
//! Op-level rather than end-to-end on purpose. A whole-model comparison tells you the
//! outputs differ; this tells you *which kernel* differs, which is the difference between a
//! day of bisecting and a minute of reading one function.

use gguf_core::{
    Activation, AttnCfg, Backend, BinOp, BufferId, DType, GluCfg, KvView, MatMulCfg, MemKind,
    NormCfg, NormKind, QuantView, RopeCfg, RopeKind,
};

/// Deterministic, reproducible, and varied enough that a transposed index shows up.
fn pseudo(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as i32 as f32 / 8388608.0) - 0.5
        })
        .collect()
}

fn pseudo_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect()
}

/// Quantized weight bytes whose scale fields are sane, so a comparison measures the kernel
/// rather than a random NaN exponent.
fn weight_bytes(dtype: DType, rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    // Float types have no block structure to patch, and random bytes there decode to NaN
    // and infinity, which measures nothing.
    if matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        let vals = pseudo(rows * cols, seed);
        return match dtype {
            DType::F32 => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
            DType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            _ => vals.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect(),
        };
    }
    let nb = rows * cols / dtype.block_size();
    let ts = dtype.type_size();
    let mut raw = pseudo_bytes(nb * ts, seed);
    let scale = half::f16::from_f32(0.0123).to_le_bytes();
    let min = half::f16::from_f32(-0.0071).to_le_bytes();
    for b in 0..nb {
        let blk = &mut raw[b * ts..(b + 1) * ts];
        match dtype {
            DType::Q4_0 | DType::Q5_0 | DType::Q8_0 | DType::Iq4Nl => blk[0..2].copy_from_slice(&scale),
            DType::Q4_1 | DType::Q5_1 | DType::Q4K | DType::Q5K => {
                blk[0..2].copy_from_slice(&scale);
                blk[2..4].copy_from_slice(&min);
            }
            DType::Q2K => {
                blk[80..82].copy_from_slice(&scale);
                blk[82..84].copy_from_slice(&min);
            }
            DType::Q3K => blk[108..110].copy_from_slice(&scale),
            DType::Q6K => blk[208..210].copy_from_slice(&scale),
            DType::Iq4Xs => blk[0..2].copy_from_slice(&scale),
            DType::Tq1_0 => blk[52..54].copy_from_slice(&scale),
            DType::Tq2_0 => blk[64..66].copy_from_slice(&scale),
            DType::Mxfp4 => blk[0] = 124,
            _ => {}
        }
    }
    raw
}

struct Pair {
    reference: Box<dyn Backend>,
    candidate: Box<dyn Backend>,
    name: String,
}

/// Every GPU backend this build can actually open, paired with a fresh CPU reference.
fn pairs() -> Vec<Pair> {
    let mut out = Vec::new();
    #[cfg(feature = "cuda")]
    if let Ok(c) = gguf_backend_cuda::CudaBackend::new(0) {
        out.push(Pair {
            reference: Box::new(gguf_backend_cpu::CpuBackend::new().unwrap()),
            name: format!("cuda ({})", c.info().name),
            candidate: Box::new(c),
        });
    }
    #[cfg(feature = "vulkan")]
    if let Ok(c) = gguf_backend_vk::VulkanBackend::new(0) {
        out.push(Pair {
            reference: Box::new(gguf_backend_cpu::CpuBackend::new().unwrap()),
            name: format!("vulkan ({})", c.info().name),
            candidate: Box::new(c),
        });
    }
    out
}

fn upload(be: &mut dyn Backend, data: &[f32]) -> BufferId {
    let b = be.alloc((data.len() * 4) as u64, MemKind::Device).unwrap();
    be.write_f32(b, data).unwrap();
    b
}

fn empty(be: &mut dyn Backend, n: usize) -> BufferId {
    let b = be.alloc((n * 4) as u64, MemKind::Device).unwrap();
    be.fill(b, 0.0, n as u32).unwrap();
    b
}

/// Compare two result vectors by normalized mean squared error.
///
/// Per-element relative error is the wrong metric here. A dot product over thousands of
/// terms produces occasional results near zero through cancellation, and a small absolute
/// difference there reads as an enormous relative one while saying nothing about whether
/// the kernel is right. NMSE measures the error against the energy of the signal, which is
/// what "these two produce the same answer" actually means.
#[track_caller]
fn assert_close(what: &str, backend: &str, a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "{what} on {backend}: length mismatch");
    let mut num = 0f64;
    let mut den = 0f64;
    let mut worst = 0f32;
    let mut at = 0usize;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if !x.is_finite() || !y.is_finite() {
            panic!("{what} on {backend}: non-finite at {i}: reference {x}, candidate {y}");
        }
        let d = (x - y).abs();
        if d > worst {
            worst = d;
            at = i;
        }
        num += (d as f64) * (d as f64);
        den += (*x as f64) * (*x as f64);
    }
    let nmse = if den > 0.0 { num / den } else { num };
    assert!(
        nmse <= tol as f64,
        "{what} on {backend}: NMSE {nmse:.3e} exceeds {tol:.1e}          (worst element {worst:.4e} at {at}: reference {}, candidate {})",
        a[at],
        b[at]
    );
}

/// Tolerances, as NMSE.
///
/// Exact ops must agree bit for bit. Anything that reduces over a long row differs by f32
/// reassociation alone. Matmul is looser still because the backends do not agree on how the
/// *activation* is represented: CPU and CUDA quantize it to Q8_1 so the dot product can run
/// in integers, while Vulkan keeps it in f32 - more accurate, but a different number. That
/// 8-bit staging is worth roughly 1e-5 of NMSE and is the floor here.
const TOL_EXACT: f32 = 1e-12;
const TOL_REDUCE: f32 = 1e-9;
const TOL_MATMUL: f32 = 1e-4;
/// A MoE layer chains three matmuls with a re-quantization between them, so whatever the
/// backends disagree about in one matmul compounds across all three.
const TOL_MOE: f32 = 1e-3;

#[test]
fn norms_match() {
    for p in pairs().iter_mut() {
        for &(dim, n_tokens) in &[(64usize, 1usize), (128, 3), (256, 2), (896, 2), (2048, 1)] {
            let src = pseudo(dim * n_tokens, 0x11 + dim as u64);
            let w = pseudo(dim, 0x22);
            for kind in [NormKind::Rms, NormKind::Layer] {
                let mut run = |be: &mut dyn Backend| {
                    let s = upload(be, &src);
                    let wb = upload(be, &w);
                    let d = empty(be, dim * n_tokens);
                    let cfg = NormCfg {
                        kind,
                        dim: dim as u32,
                        n_tokens: n_tokens as u32,
                        eps: 1e-5,
                        bias: None,
                        scale: 1.0,
                    };
                    be.norm(d, s, Some(wb), cfg).unwrap();
                    be.submit().unwrap();
                    be.read_f32(d, dim * n_tokens).unwrap()
                };
                let want = run(p.reference.as_mut());
                let got = run(p.candidate.as_mut());
                assert_close(&format!("{kind:?}norm dim={dim} n={n_tokens}"), &p.name, &want, &got, TOL_REDUCE);
            }
        }
    }
}

#[test]
fn rope_matches() {
    // head_dim 64/128/256 and both rotation layouts: the pairing and the frequency ladder
    // are the two things a backend can get subtly wrong without ever crashing.
    for p in pairs().iter_mut() {
        for &head_dim in &[64usize, 128, 256] {
            for kind in [RopeKind::Norm, RopeKind::Neox] {
                for &base in &[10000.0f32, 1000000.0] {
                    let n_heads = 4usize;
                    let n_tokens = 3usize;
                    let n = head_dim * n_heads * n_tokens;
                    let src = pseudo(n, 0x33 + head_dim as u64);
                    let positions: Vec<i32> = vec![0, 17, 1234];
                    let mut run = |be: &mut dyn Backend| {
                        let q = upload(be, &src);
                        let mut cfg = RopeCfg::new(kind, n_heads as u32, n_heads as u32, head_dim as u32, base);
                        cfg.n_tokens = n_tokens as u32;
                        be.rope(q, None, &positions, cfg).unwrap();
                        be.submit().unwrap();
                        be.read_f32(q, n).unwrap()
                    };
                    let want = run(p.reference.as_mut());
                    let got = run(p.candidate.as_mut());
                    assert_close(
                        &format!("rope {kind:?} head_dim={head_dim} base={base}"),
                        &p.name,
                        &want,
                        &got,
                        TOL_REDUCE,
                    );
                }
            }
        }
    }
}

#[test]
fn attention_matches() {
    for p in pairs().iter_mut() {
        // (head_dim, n_heads, n_kv_heads, n_tokens, kv_len, softcap, window)
        let cases: &[(usize, u32, u32, u32, u32, f32, u32)] = &[
            (64, 4, 4, 1, 16, 0.0, 0),
            (64, 8, 2, 4, 32, 0.0, 0),
            (128, 4, 4, 1, 40, 0.0, 0),
            (128, 16, 8, 3, 64, 0.0, 0),
            (256, 4, 1, 1, 48, 0.0, 0),
            (256, 8, 4, 2, 40, 50.0, 0),
            (256, 4, 1, 2, 60, 0.0, 16),
        ];
        for &(hd, nh, nkv, nt, kv_len, softcap, window) in cases {
          // Both cache storage formats. f16 is the default at run time, so it needs the
          // same scrutiny as f32 rather than being taken on faith.
          for kv_dtype in [DType::F32, DType::F16] {
            let stride = nkv as usize * hd;
            let cache_slots = 128usize;
            let q = pseudo(hd * nh as usize * nt as usize, 0x44 + hd as u64);
            let kc = pseudo(stride * cache_slots, 0x55);
            let vc = pseudo(stride * cache_slots, 0x66);
            let run = |be: &mut dyn Backend| {
                let qb = upload(be, &q);
                // Fill the cache through kv_write so the storage conversion is exercised
                // rather than bypassed by a raw upload.
                let kb = empty(be, stride * cache_slots);
                let vb = empty(be, stride * cache_slots);
                let d = empty(be, hd * nh as usize * nt as usize);
                let kv = KvView {
                    k: kb,
                    v: vb,
                    stride: stride as u32,
                    base: 0,
                    dtype: kv_dtype,
                };
                let ksrc = upload(be, &kc);
                let vsrc = upload(be, &vc);
                be.kv_write(&kv, ksrc, vsrc, 0, cache_slots as u32, stride as u32).unwrap();
                let cfg = AttnCfg {
                    n_heads: nh,
                    n_kv_heads: nkv,
                    head_dim: hd as u32,
                    n_tokens: nt,
                    kv_len,
                    start_pos: kv_len - nt,
                    scale: 1.0 / (hd as f32).sqrt(),
                    softcap,
                    window,
                    sinks: None,
                };
                be.attention(d, qb, &kv, cfg).unwrap();
                be.submit().unwrap();
                be.read_f32(d, hd * nh as usize * nt as usize).unwrap()
            };
            let want = run(p.reference.as_mut());
            let got = run(p.candidate.as_mut());
            assert_close(
                &format!(
                    "attention {kv_dtype} hd={hd} heads={nh}/{nkv} tokens={nt} kv={kv_len} cap={softcap} win={window}"
                ),
                &p.name,
                &want,
                &got,
                TOL_REDUCE,
            );
          }
        }
    }
}

#[test]
fn matmul_matches_for_every_quantization() {
    const TYPES: &[DType] = &[
        DType::Q4_0, DType::Q4_1, DType::Q5_0, DType::Q5_1, DType::Q8_0,
        DType::Q2K, DType::Q3K, DType::Q4K, DType::Q5K, DType::Q6K,
        DType::Iq4Nl, DType::Iq4Xs, DType::Mxfp4, DType::Tq1_0, DType::Tq2_0,
    ];
    for p in pairs().iter_mut() {
        for &dt in TYPES {
            let cols = if dt.block_size() == 256 { 512 } else { 256 };
            let rows = 24usize;
            for &n_tokens in &[1usize, 5] {
                let w = weight_bytes(dt, rows, cols, 0x77 ^ dt as u64);
                let act = pseudo(cols * n_tokens, 0x88);
                let mut run = |be: &mut dyn Backend| {
                    let view = QuantView::new(&w, dt, rows, cols);
                    let wid = be.upload_weight(&view).unwrap();
                    let s = upload(be, &act);
                    let d = empty(be, rows * n_tokens);
                    be.matmul(
                        d,
                        wid,
                        s,
                        MatMulCfg::new(rows as u32, cols as u32, n_tokens as u32),
                    )
                    .unwrap();
                    be.submit().unwrap();
                    be.read_f32(d, rows * n_tokens).unwrap()
                };
                let want = run(p.reference.as_mut());
                let got = run(p.candidate.as_mut());
                assert_close(
                    &format!("matmul {} tokens={n_tokens}", dt.name()),
                    &p.name,
                    &want,
                    &got,
                    TOL_MATMUL,
                );
            }
        }
    }
}

/// The same op at the shapes a real model actually uses.
///
/// The small cases above catch layout mistakes; this catches anything that only appears
/// once a row is long enough to need several passes per lane.
///
/// A genuine decode bug - a wrong shift, a misread scale - is never subtle: it moves the
/// NMSE by orders of magnitude, which is what this bound catches.
#[test]
fn matmul_matches_at_model_scale() {
    for p in pairs().iter_mut() {
        // (dtype, cols, rows, n_tokens): an attention projection, an FFN projection, and a
        // vocabulary projection.
        let cases: &[(DType, usize, usize, usize)] = &[
            (DType::Q4K, 2048, 2048, 1),
            (DType::Q4K, 2048, 6144, 13),
            (DType::Q6K, 6144, 2048, 1),
            (DType::Q5_0, 896, 4864, 7),
            (DType::Q8_0, 2048, 4096, 1),
        ];
        for &(dt, cols, rows, n_tokens) in cases {
            let w = weight_bytes(dt, rows, cols, 0xf0 ^ dt as u64 ^ cols as u64);
            let act = pseudo(cols * n_tokens, 0xf1);
            let mut run = |be: &mut dyn Backend| {
                let view = QuantView::new(&w, dt, rows, cols);
                let wid = be.upload_weight(&view).unwrap();
                let s = upload(be, &act);
                let d = empty(be, rows * n_tokens);
                be.matmul(d, wid, s, MatMulCfg::new(rows as u32, cols as u32, n_tokens as u32))
                    .unwrap();
                be.submit().unwrap();
                be.read_f32(d, rows * n_tokens).unwrap()
            };
            let want = run(p.reference.as_mut());
            let got = run(p.candidate.as_mut());
            assert_close(
                &format!("matmul {} {rows}x{cols} tokens={n_tokens}", dt.name()),
                &p.name,
                &want,
                &got,
                TOL_MATMUL,
            );
        }
    }
}

#[test]
fn get_rows_matches() {
    for p in pairs().iter_mut() {
        for &dt in &[DType::Q4K, DType::Q6K, DType::Q8_0, DType::F32] {
            let cols = if dt.block_size() == 256 { 512 } else { 256 };
            let rows = 40usize;
            let w = weight_bytes(dt, rows, cols, 0x99 ^ dt as u64);
            let tokens: Vec<u32> = vec![0, 7, 39, 12];
            let mut run = |be: &mut dyn Backend| {
                let view = QuantView::new(&w, dt, rows, cols);
                let wid = be.upload_weight(&view).unwrap();
                let d = empty(be, cols * tokens.len());
                be.get_rows(d, wid, &tokens, 2.0).unwrap();
                be.submit().unwrap();
                be.read_f32(d, cols * tokens.len()).unwrap()
            };
            let want = run(p.reference.as_mut());
            let got = run(p.candidate.as_mut());
            assert_close(&format!("get_rows {}", dt.name()), &p.name, &want, &got, TOL_REDUCE);
        }
    }
}

#[test]
fn elementwise_ops_match() {
    for p in pairs().iter_mut() {
        let n = 1024usize;
        let a = pseudo(n, 0xaa);
        let b = pseudo(n, 0xbb);

        for (label, op) in [("add", BinOp::Add), ("mul", BinOp::Mul), ("sub", BinOp::Sub)] {
            let mut run = |be: &mut dyn Backend| {
                let ab = upload(be, &a);
                let bb = upload(be, &b);
                let d = empty(be, n);
                be.binary(d, ab, bb, op, n as u32).unwrap();
                be.submit().unwrap();
                be.read_f32(d, n).unwrap()
            };
            let want = run(p.reference.as_mut());
            let got = run(p.candidate.as_mut());
            assert_close(label, &p.name, &want, &got, TOL_EXACT);
        }

        for act in [Activation::Silu, Activation::Gelu, Activation::Relu, Activation::GeluQuick] {
            let mut run = |be: &mut dyn Backend| {
                let ab = upload(be, &a);
                let bb = upload(be, &b);
                let d = empty(be, n);
                be.glu(d, ab, bb, GluCfg::new(act, n as u32, 1)).unwrap();
                be.submit().unwrap();
                be.read_f32(d, n).unwrap()
            };
            let want = run(p.reference.as_mut());
            let got = run(p.candidate.as_mut());
            assert_close(&format!("glu {act:?}"), &p.name, &want, &got, TOL_REDUCE);
        }

        let mut run = |be: &mut dyn Backend| {
            let ab = upload(be, &a);
            be.softcap(ab, 30.0, n as u32).unwrap();
            be.scale(ab, 0.5, n as u32).unwrap();
            be.submit().unwrap();
            be.read_f32(ab, n).unwrap()
        };
        let want = run(p.reference.as_mut());
        let got = run(p.candidate.as_mut());
        assert_close("softcap+scale", &p.name, &want, &got, TOL_REDUCE);
    }
}

/// Mixture-of-experts: routing, the expert matmuls, and the weighted reduction.
///
/// Worth testing synthetically rather than only through a model, because the MoE files in
/// circulation are tens of gigabytes and will not fit on the card this runs on. The kernels
/// are the same ones a 30B model would use; only the dimensions are smaller.
#[test]
fn moe_matches() {
    use gguf_core::{GateFunc, MoeCfg, MoeWeights};

    for p in pairs().iter_mut() {
        let n_embd = 256usize;
        let n_ff = 512usize;
        let n_expert = 8usize;
        let n_used = 2usize;

        for &n_tokens in &[1usize, 4] {
            for gate_func in [GateFunc::Softmax, GateFunc::Sigmoid] {
                for &norm_topk in &[true, false] {
                    let router = weight_bytes(DType::Q8_0, n_expert, n_embd, 0x201);
                    // Experts are stacked: [n_expert * rows, cols].
                    let gate = weight_bytes(DType::Q4K, n_expert * n_ff, n_embd, 0x202);
                    let up = weight_bytes(DType::Q4K, n_expert * n_ff, n_embd, 0x203);
                    let down = weight_bytes(DType::Q6K, n_expert * n_embd, n_ff, 0x204);
                    let act = pseudo(n_embd * n_tokens, 0x205);

                    let mut run = |be: &mut dyn Backend| {
                        let w = MoeWeights {
                            gate_inp: be
                                .upload_weight(&QuantView::new(&router, DType::Q8_0, n_expert, n_embd))
                                .unwrap(),
                            gate_exps: Some(
                                be.upload_weight(&QuantView::new(
                                    &gate, DType::Q4K, n_expert * n_ff, n_embd,
                                ))
                                .unwrap(),
                            ),
                            up_exps: be
                                .upload_weight(&QuantView::new(&up, DType::Q4K, n_expert * n_ff, n_embd))
                                .unwrap(),
                            down_exps: be
                                .upload_weight(&QuantView::new(
                                    &down, DType::Q6K, n_expert * n_embd, n_ff,
                                ))
                                .unwrap(),
                            exp_probs_b: None,
                        };
                        let src = upload(be, &act);
                        let d = empty(be, n_embd * n_tokens);
                        be.moe(
                            d,
                            src,
                            &w,
                            MoeCfg {
                                n_expert: n_expert as u32,
                                n_expert_used: n_used as u32,
                                n_embd: n_embd as u32,
                                n_ff: n_ff as u32,
                                n_tokens: n_tokens as u32,
                                act: Activation::Silu,
                                gate_func,
                                norm_topk,
                                scale: 1.0,
                            },
                        )
                        .unwrap();
                        be.submit().unwrap();
                        be.read_f32(d, n_embd * n_tokens).unwrap()
                    };
                    let want = run(p.reference.as_mut());
                    let got = run(p.candidate.as_mut());
                    assert_close(
                        &format!("moe {gate_func:?} norm_topk={norm_topk} tokens={n_tokens}"),
                        &p.name,
                        &want,
                        &got,
                        TOL_MOE,
                    );
                }
            }
        }
    }
}

#[test]
fn kv_write_then_read_matches() {
    for p in pairs().iter_mut() {
        let dim = 128usize;
        let slots = 64usize;
        let n_tokens = 3usize;
        let k = pseudo(dim * n_tokens, 0xcc);
        let v = pseudo(dim * n_tokens, 0xdd);
        for dtype in [DType::F32, DType::F16] {
            let words = dim * slots * dtype.type_size() / 4;
            let run = |be: &mut dyn Backend| {
                let kb = upload(be, &k);
                let vb = upload(be, &v);
                let kc = empty(be, words);
                let vc = empty(be, words);
                let kv = KvView { k: kc, v: vc, stride: dim as u32, base: 8, dtype };
                be.kv_write(&kv, kb, vb, 5, n_tokens as u32, dim as u32).unwrap();
                be.submit().unwrap();
                // Compared as raw words: the backends must agree on the stored bits, not
                // merely on what they decode to.
                let mut out = be.read_f32(kc, words).unwrap();
                out.extend(be.read_f32(vc, words).unwrap());
                out.iter().map(|v| v.to_bits() as f32).collect::<Vec<f32>>()
            };
            let want = run(p.reference.as_mut());
            let got = run(p.candidate.as_mut());
            assert_close(&format!("kv_write {dtype}"), &p.name, &want, &got, TOL_EXACT);
        }
    }
}

/// Diagnostic sweep: report, rather than assert, how the error scales with row length.
/// Run with `--ignored --nocapture` when a backend disagreement needs localizing.
#[test]
#[ignore]
fn sweep_matmul_shapes() {
    for p in pairs().iter_mut() {
        for &dt in &[DType::Q4_0, DType::Q8_0, DType::Q4K, DType::Q6K] {
            for &cols in &[256usize, 512, 1024, 2048, 4096, 6144] {
                if cols % dt.block_size() != 0 {
                    continue;
                }
                let rows = 8usize;
                let w = weight_bytes(dt, rows, cols, 0xabc ^ dt as u64 ^ cols as u64);
                let act = pseudo(cols, 0xdef);
                let run = |be: &mut dyn Backend| {
                    let wid = be.upload_weight(&QuantView::new(&w, dt, rows, cols)).unwrap();
                    let s = upload(be, &act);
                    let d = empty(be, rows);
                    be.matmul(d, wid, s, MatMulCfg::new(rows as u32, cols as u32, 1)).unwrap();
                    be.submit().unwrap();
                    be.read_f32(d, rows).unwrap()
                };
                let a = run(p.reference.as_mut());
                let b = run(p.candidate.as_mut());
                let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                let mag = a.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1e-6);
                println!(
                    "{:<8} {:<6} cols={:<6} worst_abs={:.3e} rel={:.3e}  cpu[0]={:+.5} gpu[0]={:+.5}",
                    p.name.split(' ').next().unwrap(), dt.name(), cols, worst, worst / mag, a[0], b[0]
                );
            }
        }
    }
}
