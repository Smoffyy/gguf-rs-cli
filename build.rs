use std::path::PathBuf;
use naga::front::glsl;
use naga::back::spv;
use naga::valid::{Validator, ValidationFlags, Capabilities};

fn compile(src: &str, out_name: &str, out_dir: &str) {
    let src_text = std::fs::read_to_string(src)
        .unwrap_or_else(|_| panic!("Cannot read {}", src));
    let mut parser = glsl::Frontend::default();
    let opts  = glsl::Options::from(naga::ShaderStage::Compute);
    let module = parser.parse(&opts, &src_text)
        .unwrap_or_else(|e| panic!("Parse {}: {:?}", src, e));
    let mut v  = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info   = v.validate(&module)
        .unwrap_or_else(|e| panic!("Validate {}: {:?}", src, e));
    let spv_opts = spv::Options {
        lang_version: (1, 0),
        flags: spv::WriterFlags::empty(),
        capabilities: None,
        bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
        zero_initialize_workgroup_memory: spv::ZeroInitializeWorkgroupMemoryMode::None,
        debug_info: None,
        binding_map: std::collections::BTreeMap::new(),
    };
    let words = spv::write_vec(&module, &info, &spv_opts, None)
        .unwrap_or_else(|e| panic!("SPV {}: {:?}", src, e));
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let out = PathBuf::from(out_dir).join(format!("{}.spv", out_name));
    std::fs::write(&out, &bytes).unwrap();
    println!("cargo:rerun-if-changed={}", src);
}

fn main() {
    let out = std::env::var("OUT_DIR").unwrap();
    println!("cargo:rerun-if-changed=build.rs");
    for (src, name) in &[
        ("src/gpu/q4_0_gemv.glsl",  "q4_0_gemv"),
        ("src/gpu/q4k_gemv.glsl",   "q4k_gemv"),
        ("src/gpu/q6k_gemv.glsl",   "q6k_gemv"),
        ("src/gpu/q8_0_gemv.glsl",  "q8_0_gemv"),
        ("src/gpu/f32_gemv.glsl",   "f32_gemv"),
        ("src/gpu/rmsnorm.glsl",    "rmsnorm"),
        ("src/gpu/rope.glsl",       "rope"),
        ("src/gpu/kv_write.glsl",   "kv_write"),
        ("src/gpu/attention.glsl",  "attention"),
        ("src/gpu/swiglu.glsl",     "swiglu"),
        ("src/gpu/add.glsl",        "add"),
    ] { compile(src, name, &out); }
}