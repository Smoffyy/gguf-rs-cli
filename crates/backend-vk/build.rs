//! Compile the GLSL compute shaders to SPIR-V and embed them.
//!
//! `#include "common.glsl"` is resolved here by textual substitution rather than through a
//! shaderc include callback: there is exactly one include in the whole shader set, and
//! doing it this way keeps the shaders readable as standalone files while leaving nothing
//! to configure.

use std::path::PathBuf;

const SHADERS: &[&str] = &[
    "matmul",
    "get_rows",
    "norm",
    "rope",
    "kv_write",
    "attention",
    "attn_combine",
    "elementwise",
    "moe_route",
    "moe_mm",
    "moe_misc",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shaders/common.glsl");
    for s in SHADERS {
        println!("cargo:rerun-if-changed=shaders/{s}.comp");
    }

    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let common = std::fs::read_to_string("shaders/common.glsl")
        .expect("shaders/common.glsl is missing");

    let compiler = shaderc::Compiler::new().expect("shaderc is unavailable");
    let mut options = shaderc::CompileOptions::new().expect("shaderc options");
    // Vulkan 1.0 keeps the widest driver compatibility, which is the point of having a
    // Vulkan backend alongside CUDA at all.
    options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_0 as u32);
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);

    for name in SHADERS {
        let path = format!("shaders/{name}.comp");
        let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        // The include sits after the interface declarations because `common.glsl` refers
        // to the WEIGHTS binding that each shader declares for itself.
        let expanded = src.replace("#include \"common.glsl\"", &common);

        let artifact = compiler
            .compile_into_spirv(
                &expanded,
                shaderc::ShaderKind::Compute,
                &path,
                "main",
                Some(&options),
            )
            .unwrap_or_else(|e| panic!("compiling {path}:\n{e}"));

        std::fs::write(out.join(format!("{name}.spv")), artifact.as_binary_u8()).unwrap();
    }
}
