//! Compile the CUDA kernels to PTX at build time.
//!
//! PTX rather than a cubin, and loaded through the driver rather than linked: the driver
//! JIT-compiles PTX for whatever GPU is actually present, so one build covers every
//! architecture from the chosen floor upward without a fat binary and without the CUDA
//! runtime being a link-time dependency.
//!
//! If nvcc is not installed the backend still compiles - it just reports itself
//! unavailable at run time. A missing optional toolchain should not break the build of an
//! engine whose whole point is that the CPU path always works.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Virtual architectures to try, newest first. CUDA 13 dropped Maxwell and Pascal, older
/// toolkits do not know Ada, so the first one nvcc accepts wins.
const ARCHES: &[&str] = &["compute_75", "compute_70", "compute_61"];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/ops.cu");
    println!("cargo:rerun-if-changed=kernels/quants.cuh");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=GGUF_RS_NO_CUDA");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ops.ptx");

    if std::env::var("GGUF_RS_NO_CUDA").is_ok() {
        disable(&ptx_path, "GGUF_RS_NO_CUDA is set");
        return;
    }

    let Some(nvcc) = find_nvcc() else {
        disable(
            &ptx_path,
            "nvcc was not found (set CUDA_PATH or put nvcc on PATH to enable the CUDA backend)",
        );
        return;
    };

    // On Windows nvcc drives MSVC as its host compiler even for PTX-only output, and it
    // refuses host toolchains newer than it knows about. Several MSVC versions are usually
    // installed side by side, so try each rather than guessing.
    let hosts = host_compilers();
    let mut last = String::from("no attempt was made");
    for host in &hosts {
        for arch in ARCHES {
            match compile(&nvcc, arch, host.as_deref(), &ptx_path) {
                Ok(()) => {
                    println!("cargo:rustc-cfg=cuda_kernels");
                    println!("cargo:warning=CUDA kernels compiled for {arch}");
                    return;
                }
                Err(e) => last = e,
            }
        }
    }
    disable(&ptx_path, &format!("nvcc could not build the kernels: {last}"));
}

fn compile(nvcc: &Path, arch: &str, ccbin: Option<&Path>, out: &Path) -> Result<(), String> {
    let mut cmd = Command::new(nvcc);
    cmd.arg("--ptx")
        .arg(format!("-arch={arch}"))
        .arg("-O3")
        .arg("-o")
        .arg(out)
        .arg("kernels/ops.cu");
    if let Some(cc) = ccbin {
        cmd.arg("-ccbin").arg(cc);
    }
    let status = cmd.output().map_err(|e| format!("could not run nvcc: {e}"))?;
    if status.status.success() && out.exists() && out.metadata().map(|m| m.len()).unwrap_or(0) > 0 {
        return Ok(());
    }
    let err = String::from_utf8_lossy(&status.stderr);
    // Keep the first real error line; nvcc is verbose about warnings.
    Err(err
        .lines()
        .find(|l| l.contains("error") || l.contains("fatal"))
        .unwrap_or_else(|| err.trim())
        .trim()
        .to_string())
}

/// Candidate host-compiler directories, most likely first. `None` means "whatever is on
/// PATH", which is the right answer on Linux and inside a Developer Command Prompt.
fn host_compilers() -> Vec<Option<PathBuf>> {
    if !cfg!(windows) {
        return vec![None];
    }
    let mut found: Vec<PathBuf> = Vec::new();
    let roots = [
        "C:/Program Files/Microsoft Visual Studio",
        "C:/Program Files (x86)/Microsoft Visual Studio",
    ];
    for root in roots {
        let Ok(years) = std::fs::read_dir(root) else { continue };
        for year in years.flatten() {
            let Ok(editions) = std::fs::read_dir(year.path()) else { continue };
            for edition in editions.flatten() {
                let msvc = edition.path().join("VC/Tools/MSVC");
                let Ok(versions) = std::fs::read_dir(&msvc) else { continue };
                for v in versions.flatten() {
                    let bin = v.path().join("bin/Hostx64/x64");
                    if bin.join("cl.exe").exists() {
                        found.push(bin);
                    }
                }
            }
        }
    }
    // Newest first, but every one gets tried: the newest MSVC is often too new for the
    // installed CUDA, and the next one down works.
    found.sort();
    found.reverse();
    let mut out: Vec<Option<PathBuf>> = vec![None];
    out.extend(found.into_iter().map(Some));
    out
}

/// Write an empty PTX so `include_str!` still resolves, and leave the `cuda_kernels` cfg
/// unset so the backend knows to report itself unavailable.
fn disable(ptx_path: &Path, why: &str) {
    let _ = std::fs::write(ptx_path, "");
    println!("cargo:warning=CUDA backend disabled: {why}");
}

fn find_nvcc() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };

    if let Ok(root) = std::env::var("CUDA_PATH") {
        let p = PathBuf::from(root).join("bin").join(exe);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(root) = std::env::var("CUDA_HOME") {
        let p = PathBuf::from(root).join("bin").join(exe);
        if p.exists() {
            return Some(p);
        }
    }

    // Toolkits install side by side; prefer the newest.
    let roots: &[&str] = if cfg!(windows) {
        &["C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"]
    } else {
        &["/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda"]
    };
    let mut best: Option<(String, PathBuf)> = None;
    for root in roots {
        let direct = PathBuf::from(root).join("bin").join(exe);
        if direct.exists() {
            return Some(direct);
        }
        let Ok(entries) = std::fs::read_dir(root) else { continue };
        for e in entries.flatten() {
            let candidate = e.path().join("bin").join(exe);
            if candidate.exists() {
                let version = e.file_name().to_string_lossy().into_owned();
                if best.as_ref().map_or(true, |(v, _)| version > *v) {
                    best = Some((version, candidate));
                }
            }
        }
    }
    if let Some((_, p)) = best {
        return Some(p);
    }

    // Last resort: whatever is on PATH.
    Command::new(exe)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|_| PathBuf::from(exe))
}
