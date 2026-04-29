struct Params { rows: u32, bpr: u32 }

@group(0) @binding(0) var<storage, read>       matrix:  array<u32>;
@group(0) @binding(1) var<storage, read>       vec_in:  array<f32>;
@group(0) @binding(2) var<storage, read_write> vec_out: array<f32>;
@group(0) @binding(3) var<uniform>             params:  Params;

// ── Q4_0 GEMV ────────────────────────────────────────────────────────────────
// Block: 5 u32s (32 weights)
//   u32[0]     = scale as f32 bits  (pre-converted on CPU)
//   u32[1..4]  = 16 nibble-bytes packed as 4 u32s
// Layout: lower nibble of byte j → element j, upper nibble → element j+16
@compute @workgroup_size(64)
fn q4_0_gemv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.rows { return; }
    var sum = 0.0f;
    for (var b = 0u; b < params.bpr; b++) {
        let base  = (row * params.bpr + b) * 5u;
        let scale = bitcast<f32>(matrix[base]);
        let vb    = b * 32u;
        for (var w = 0u; w < 4u; w++) {
            let packed = matrix[base + 1u + w];
            for (var i = 0u; i < 4u; i++) {
                let byte   = (packed >> (i * 8u)) & 0xFFu;
                let j      = w * 4u + i;         // byte index 0..15
                sum += f32(i32(byte & 0xFu) - 8) * scale * vec_in[vb + j];
                sum += f32(i32(byte >>  4u) - 8) * scale * vec_in[vb + j + 16u];
            }
        }
    }
    vec_out[row] = sum;
}

// ── Q4_K GEMV ────────────────────────────────────────────────────────────────
// Block: 48 u32s (256 weights), pre-processed on CPU:
//   u32[0..7]   = 8× (df * scale[i]) as f32 bits
//   u32[8..15]  = 8× (dmin * min[i]) as f32 bits
//   u32[16..47] = 32 u32s of nibbles (128 bytes)
// 4 iterations of 64 elements each:
//   iter i: lo-nibbles → elements [i*64 .. i*64+32], scale=u32[i*2], min=u32[8+i*2]
//            hi-nibbles → elements [i*64+32 .. i*64+64], scale=u32[i*2+1], min=u32[9+i*2]
@compute @workgroup_size(64)
fn q4k_gemv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.rows { return; }
    var sum = 0.0f;
    for (var b = 0u; b < params.bpr; b++) {
        let base = (row * params.bpr + b) * 48u;
        let vb   = b * 256u;
        for (var iter = 0u; iter < 4u; iter++) {
            let d1 = bitcast<f32>(matrix[base + iter * 2u]);
            let d2 = bitcast<f32>(matrix[base + iter * 2u + 1u]);
            let m1 = bitcast<f32>(matrix[base + 8u + iter * 2u]);
            let m2 = bitcast<f32>(matrix[base + 8u + iter * 2u + 1u]);
            let qs_off = base + 16u + iter * 8u;
            let v_off  = vb + iter * 64u;
            for (var l = 0u; l < 8u; l++) {
                let packed = matrix[qs_off + l];
                for (var i = 0u; i < 4u; i++) {
                    let byte   = (packed >> (i * 8u)) & 0xFFu;
                    let lo_idx = v_off + l * 4u + i;
                    sum += (d1 * f32(byte & 0xFu) - m1) * vec_in[lo_idx];
                    sum += (d2 * f32(byte >>  4u) - m2) * vec_in[lo_idx + 32u];
                }
            }
        }
    }
    vec_out[row] = sum;
}

// ── Q6_K GEMV ────────────────────────────────────────────────────────────────
// Block: 64 u32s (256 weights), pre-processed on CPU:
//   u32[0..15]  = 16× (df * sc[i]) as f32 bits
//   u32[16..47] = 32 u32s of ql (128 bytes, lower 4 bits of each 6-bit value)
//   u32[48..63] = 16 u32s of qh (64 bytes, upper 2 bits)
// Two halves of 128 elements. For each half:
//   scale for output [y+l]    = sc[si + l/16]
//   scale for output [y+l+32] = sc[si + l/16 + 2]
//   scale for output [y+l+64] = sc[si + l/16 + 4]
//   scale for output [y+l+96] = sc[si + l/16 + 6]
@compute @workgroup_size(64)
fn q6k_gemv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.rows { return; }
    var sum = 0.0f;
    for (var b = 0u; b < params.bpr; b++) {
        let base = (row * params.bpr + b) * 64u;
        let vb   = b * 256u;
        for (var half = 0u; half < 2u; half++) {
            // ql: u32[16..47], each half is 64 bytes = 16 u32s
            let ql_off = base + 16u + half * 16u;
            // qh: u32[48..63], each half is 32 bytes = 8 u32s
            let qh_off = base + 48u + half * 8u;
            let y_base = vb + half * 128u;
            let si     = half * 8u;
            for (var l = 0u; l < 32u; l++) {
                // ql byte at l and l+32 within this half
                let ql_lo = (matrix[ql_off + l / 4u]       >> ((l % 4u) * 8u))       & 0xFFu;
                let ql_hi = (matrix[ql_off + (l+32u) / 4u] >> (((l+32u) % 4u) * 8u)) & 0xFFu;
                let qh_b  = (matrix[qh_off + l / 4u]       >> ((l % 4u) * 8u))       & 0xFFu;
                // Reconstruct 6-bit signed values, bias -32
                let q1 = f32(i32((ql_lo & 0xFu) | ((qh_b & 3u)        << 4u)) - 32);
                let q2 = f32(i32((ql_hi & 0xFu) | (((qh_b >> 2u) & 3u) << 4u)) - 32);
                let q3 = f32(i32((ql_lo >> 4u)  | (((qh_b >> 4u) & 3u) << 4u)) - 32);
                let q4 = f32(i32((ql_hi >> 4u)  | (((qh_b >> 6u) & 3u) << 4u)) - 32);
                let is  = l / 16u;
                let sc0 = bitcast<f32>(matrix[base + si + is]);
                let sc2 = bitcast<f32>(matrix[base + si + is + 2u]);
                let sc4 = bitcast<f32>(matrix[base + si + is + 4u]);
                let sc6 = bitcast<f32>(matrix[base + si + is + 6u]);
                sum += sc0 * q1 * vec_in[y_base + l];
                sum += sc2 * q2 * vec_in[y_base + l + 32u];
                sum += sc4 * q3 * vec_in[y_base + l + 64u];
                sum += sc6 * q4 * vec_in[y_base + l + 96u];
            }
        }
    }
    vec_out[row] = sum;
}

// ── Q8_0 GEMV ────────────────────────────────────────────────────────────────
// Block: 9 u32s (32 weights), pre-processed on CPU:
//   u32[0]    = d as f32 bits (pre-converted from f16)
//   u32[1..8] = 32 signed i8 values packed as 8 u32s
@compute @workgroup_size(64)
fn q8_0_gemv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.rows { return; }
    var sum = 0.0f;
    for (var b = 0u; b < params.bpr; b++) {
        let base = (row * params.bpr + b) * 9u;
        let d    = bitcast<f32>(matrix[base]);
        let vb   = b * 32u;
        for (var l = 0u; l < 8u; l++) {
            let packed = matrix[base + 1u + l];
            for (var i = 0u; i < 4u; i++) {
                let byte = (packed >> (i * 8u)) & 0xFFu;
                // Interpret as signed byte: >= 128 means negative
                let q    = select(f32(i32(byte) - 256), f32(byte), byte < 128u);
                sum     += d * q * vec_in[vb + l * 4u + i];
            }
        }
    }
    vec_out[row] = sum;
}

// ── F32 GEMV (fallback) ───────────────────────────────────────────────────────
// Matrix stored as raw f32 values. bpr = cols.
@compute @workgroup_size(64)
fn f32_gemv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.rows { return; }
    var sum = 0.0f;
    for (var c = 0u; c < params.bpr; c++) {
        sum += bitcast<f32>(matrix[row * params.bpr + c]) * vec_in[c];
    }
    vec_out[row] = sum;
}