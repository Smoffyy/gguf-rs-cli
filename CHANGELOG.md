# Changelog
All notable changes will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-05-03
### Added
Add suport for more quantization types.

src/gpu/q4_1_gemv.glsl — Q4_1 shader (4-bit asymmetric with scale + min per 32-weight block)
src/gpu/q3k_gemv.glsl — Q3_K shader (3-bit with high-bit mask, 6-bit scales, 256-weight superblocks)
src/gpu/q5k_gemv.glsl — Q5_K shader (5-bit with high bits, scale + min, 256-weight superblocks)

### Changed
src/tensor/dequant.rs — added pack_q4_1_for_gpu(), pack_q3k_for_gpu(), pack_q5k_for_gpu()
src/gpu/mod.rs (gpu_mod.rs) — added Q4_1, Q3K, Q5K to shader enum, pipeline creation, and upload dispatch
build.rs — added the 3 new shaders to the compile list


## [1.0.0] - 2026-05-03
### Added
First initial release, will contain bugs but is functional!