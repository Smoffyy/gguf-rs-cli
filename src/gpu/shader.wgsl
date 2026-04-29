struct Params { rows: u32, bpr: u32 }

@group(0) @binding(0) var<storage, read> matrix_q4: array<u32>;
@group(0) @binding(1) var<storage, read> vec_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> vec_out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// Q4_0 GEMV: matrix stored as 5 u32s per block
// [f32_scale_bits, nibble_u32 x4]  (32 weights per block)
@compute @workgroup_size(64)
fn q4_gemv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.rows { return; }

    var sum = 0.0f;
    let bpr = params.bpr;

    for (var b = 0u; b < bpr; b++) {
        let base = (row * bpr + b) * 5u;
        let scale = bitcast<f32>(matrix_q4[base]);
        let vb = b * 32u;

        for (var w = 0u; w < 4u; w++) {
            let packed = matrix_q4[base + 1u + w];
            for (var i = 0u; i < 4u; i++) {
                let byte  = (packed >> (i * 8u)) & 0xFFu;
                let lo    = f32(i32(byte & 0xFu) - 8) * scale;
                let hi    = f32(i32(byte >> 4u) - 8) * scale;
                let idx   = vb + w * 8u + i * 2u;
                sum += lo * vec_in[idx];
                sum += hi * vec_in[idx + 1u];
            }
        }
    }
    vec_out[row] = sum;
}

// F32 GEMV fallback — reuses same binding layout, matrix_q4 reinterpreted as f32
@compute @workgroup_size(64)
fn f32_gemv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.rows { return; }
    var sum = 0.0f;
    let cols = params.bpr;
    for (var c = 0u; c < cols; c++) {
        sum += bitcast<f32>(matrix_q4[row * cols + c]) * vec_in[c];
    }
    vec_out[row] = sum;
}