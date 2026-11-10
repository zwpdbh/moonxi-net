// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
#borrow(input, output, metadata)
extern "c" fn moonbit_bf16_direct_layout(
  input : GpuBuffer,
  output : GpuBuffer,
  metadata : GpuBuffer,
  n : Int32,
) -> Int32 = "moonbit_bf16_direct_layout"

///|
pub fn bf16_direct_layout(
  input : GpuBuffer,
  output : GpuBuffer,
  metadata : GpuBuffer,
  n : Int32,
) -> Int32 {
  moonbit_bf16_direct_layout(input, output, metadata, n)
}

///|
#borrow(input, indices, output)
extern "c" fn moonbit_bf16_gather(
  input : GpuBuffer,
  indices : GpuBuffer,
  output : GpuBuffer,
  n : Int32,
) -> Int32 = "moonbit_bf16_gather"

///|
pub fn bf16_gather(
  input : GpuBuffer,
  indices : GpuBuffer,
  output : GpuBuffer,
  n : Int32,
) -> Int32 {
  moonbit_bf16_gather(input, indices, output, n)
}

///|
#borrow(left, right, output)
extern "c" fn moonbit_bf16_concat2(
  left : GpuBuffer,
  right : GpuBuffer,
  output : GpuBuffer,
  outer : Int32,
  left_width : Int32,
  right_width : Int32,
  inner : Int32,
) -> Int32 = "moonbit_bf16_concat2"

///|
pub fn bf16_concat2(
  left : GpuBuffer,
  right : GpuBuffer,
  output : GpuBuffer,
  outer : Int32,
  left_width : Int32,
  right_width : Int32,
  inner : Int32,
) -> Int32 {
  moonbit_bf16_concat2(
    left, right, output, outer, left_width, right_width, inner,
  )
}

///|
#borrow(handle, left, right, output)
extern "c" fn moonbit_bf16_batched_matmul(
  handle : CublasHandle,
  left : GpuBuffer,
  right : GpuBuffer,
  output : GpuBuffer,
  batch : Int32,
  m : Int32,
  k : Int32,
  p : Int32,
  transpose_right : Int32,
) -> Int32 = "moonbit_bf16_batched_matmul"

///|
pub fn bf16_batched_matmul(
  handle : CublasHandle,
  left : GpuBuffer,
  right : GpuBuffer,
  output : GpuBuffer,
  batch : Int32,
  m : Int32,
  k : Int32,
  p : Int32,
  transpose_right : Int32,
) -> Int32 {
  moonbit_bf16_batched_matmul(
    handle, left, right, output, batch, m, k, p, transpose_right,
  )
}

///|
#borrow(handle, left, right, output)
extern "c" fn moonbit_bf16_linear(
  handle : CublasHandle,
  left : GpuBuffer,
  right : GpuBuffer,
  output : GpuBuffer,
  rows : Int32,
  input_size : Int32,
  output_size : Int32,
) -> Int32 = "moonbit_bf16_linear"

///|
pub fn bf16_linear(
  handle : CublasHandle,
  left : GpuBuffer,
  right : GpuBuffer,
  output : GpuBuffer,
  rows : Int32,
  input_size : Int32,
  output_size : Int32,
) -> Int32 {
  moonbit_bf16_linear(
    handle, left, right, output, rows, input_size, output_size,
  )
}

///|
#borrow(input, weight, output)
extern "c" fn moonbit_bf16_rms_norm(
  input : GpuBuffer,
  weight : GpuBuffer,
  output : GpuBuffer,
  width : Int32,
  rows : Int32,
  eps : Float,
) -> Int32 = "moonbit_bf16_rms_norm"

///|
pub fn bf16_rms_norm(
  input : GpuBuffer,
  weight : GpuBuffer,
  output : GpuBuffer,
  width : Int32,
  rows : Int32,
  eps : Float,
) -> Int32 {
  moonbit_bf16_rms_norm(input, weight, output, width, rows, eps)
}

///|
#borrow(input, output)
extern "c" fn moonbit_bf16_softmax(
  input : GpuBuffer,
  output : GpuBuffer,
  width : Int32,
  rows : Int32,
) -> Int32 = "moonbit_bf16_softmax"

///|
pub fn bf16_softmax(
  input : GpuBuffer,
  output : GpuBuffer,
  width : Int32,
  rows : Int32,
) -> Int32 {
  moonbit_bf16_softmax(input, output, width, rows)
}

///|
#borrow(input, output)
extern "c" fn moonbit_bf16_silu(
  input : GpuBuffer,
  output : GpuBuffer,
  n : Int32,
) -> Int32 = "moonbit_bf16_silu"

///|
pub fn bf16_silu(input : GpuBuffer, output : GpuBuffer, n : Int32) -> Int32 {
  moonbit_bf16_silu(input, output, n)
}

///|
#borrow(left, right, output)
extern "c" fn moonbit_bf16_mul(
  left : GpuBuffer,
  right : GpuBuffer,
  output : GpuBuffer,
  n : Int32,
  right_size : Int32,
) -> Int32 = "moonbit_bf16_mul"

///|
pub fn bf16_mul(
  left : GpuBuffer,
  right : GpuBuffer,
  output : GpuBuffer,
  n : Int32,
  right_size : Int32,
) -> Int32 {
  moonbit_bf16_mul(left, right, output, n, right_size)
}

///|
#borrow(left, right, output)
extern "c" fn moonbit_bf16_add(
  left : GpuBuffer,
  right : GpuBuffer,
  output : GpuBuffer,
  n : Int32,
) -> Int32 = "moonbit_bf16_add"

///|
pub fn bf16_add(
  left : GpuBuffer,
  right : GpuBuffer,
  output : GpuBuffer,
  n : Int32,
) -> Int32 {
  moonbit_bf16_add(left, right, output, n)
}

///|
#borrow(input, output)
extern "c" fn moonbit_bf16_scale(
  input : GpuBuffer,
  output : GpuBuffer,
  n : Int32,
  scalar : Float,
) -> Int32 = "moonbit_bf16_scale"

///|
pub fn bf16_scale(
  input : GpuBuffer,
  output : GpuBuffer,
  n : Int32,
  scalar : Float,
) -> Int32 {
  moonbit_bf16_scale(input, output, n, scalar)
}

///|
#borrow(input, mask, output)
extern "c" fn moonbit_bf16_masked_fill(
  input : GpuBuffer,
  mask : GpuBuffer,
  output : GpuBuffer,
  n : Int32,
  mask_size : Int32,
  value : Float,
) -> Int32 = "moonbit_bf16_masked_fill"

///|
pub fn bf16_masked_fill(
  input : GpuBuffer,
  mask : GpuBuffer,
  output : GpuBuffer,
  n : Int32,
  mask_size : Int32,
  value : Float,
) -> Int32 {
  moonbit_bf16_masked_fill(input, mask, output, n, mask_size, value)
}

///|
#borrow(input, output)
extern "c" fn moonbit_bf16_argmax(
  input : GpuBuffer,
  output : GpuBuffer,
  rows : Int32,
  width : Int32,
) -> Int32 = "moonbit_bf16_argmax"

///|
pub fn bf16_argmax(
  input : GpuBuffer,
  output : GpuBuffer,
  rows : Int32,
  width : Int32,
) -> Int32 {
  moonbit_bf16_argmax(input, output, rows, width)
}

///|
#borrow(input, output)
extern "c" fn moonbit_bf16_to_host(
  input : GpuBuffer,
  output : FixedArray[Float],
  n : Int32,
) -> Int32 = "moonbit_bf16_to_host"

///|
pub fn bf16_to_host(
  input : GpuBuffer,
  output : FixedArray[Float],
  n : Int32,
) -> Int32 {
  moonbit_bf16_to_host(input, output, n)
}
