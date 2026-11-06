// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
#borrow(input, indices, output)
extern "c" fn moonbit_cuda_llm_gather(
  input : GpuBuffer,
  indices : GpuBuffer,
  output : GpuBuffer,
  n : Int32,
) -> Int32 = "moonbit_cuda_llm_gather"

///|
pub fn llm_gather(
  input : GpuBuffer,
  indices : GpuBuffer,
  output : GpuBuffer,
  n : Int32,
) -> Int32 {
  moonbit_cuda_llm_gather(input, indices, output, n)
}

///|
#borrow(handle, left, right, output)
extern "c" fn moonbit_cuda_llm_batched_matmul(
  handle : CublasHandle,
  left : GpuBuffer,
  right : GpuBuffer,
  output : GpuBuffer,
  batch : Int32,
  m : Int32,
  k : Int32,
  p : Int32,
  transpose_right : Int32,
) -> Int32 = "moonbit_cuda_llm_batched_matmul"

///|
pub fn llm_batched_matmul(
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
  moonbit_cuda_llm_batched_matmul(
    handle, left, right, output, batch, m, k, p, transpose_right,
  )
}

///|
#borrow(input, output)
extern "c" fn moonbit_cuda_llm_softmax(
  input : GpuBuffer,
  output : GpuBuffer,
  width : Int32,
  rows : Int32,
) -> Int32 = "moonbit_cuda_llm_softmax"

///|
pub fn llm_softmax(
  input : GpuBuffer,
  output : GpuBuffer,
  width : Int32,
  rows : Int32,
) -> Int32 {
  moonbit_cuda_llm_softmax(input, output, width, rows)
}

///|
#borrow(input, output)
extern "c" fn moonbit_cuda_llm_silu(
  input : GpuBuffer,
  output : GpuBuffer,
  n : Int32,
) -> Int32 = "moonbit_cuda_llm_silu"

///|
pub fn llm_silu(input : GpuBuffer, output : GpuBuffer, n : Int32) -> Int32 {
  moonbit_cuda_llm_silu(input, output, n)
}

///|
#borrow(input, weight, output)
extern "c" fn moonbit_cuda_llm_rms_norm(
  input : GpuBuffer,
  weight : GpuBuffer,
  output : GpuBuffer,
  width : Int32,
  rows : Int32,
  eps : Float,
) -> Int32 = "moonbit_cuda_llm_rms_norm"

///|
pub fn llm_rms_norm(
  input : GpuBuffer,
  weight : GpuBuffer,
  output : GpuBuffer,
  width : Int32,
  rows : Int32,
  eps : Float,
) -> Int32 {
  moonbit_cuda_llm_rms_norm(input, weight, output, width, rows, eps)
}

///|
#borrow(input, mask, output)
extern "c" fn moonbit_cuda_llm_masked_fill(
  input : GpuBuffer,
  mask : GpuBuffer,
  output : GpuBuffer,
  n : Int32,
  mask_size : Int32,
  value : Float,
) -> Int32 = "moonbit_cuda_llm_masked_fill"

///|
pub fn llm_masked_fill(
  input : GpuBuffer,
  mask : GpuBuffer,
  output : GpuBuffer,
  n : Int32,
  mask_size : Int32,
  value : Float,
) -> Int32 {
  moonbit_cuda_llm_masked_fill(input, mask, output, n, mask_size, value)
}

///|
#borrow(left, right, output)
extern "c" fn moonbit_cuda_llm_mul_broadcast(
  left : GpuBuffer,
  right : GpuBuffer,
  output : GpuBuffer,
  n : Int32,
  right_size : Int32,
) -> Int32 = "moonbit_cuda_llm_mul_broadcast"

///|
pub fn llm_mul_broadcast(
  left : GpuBuffer,
  right : GpuBuffer,
  output : GpuBuffer,
  n : Int32,
  right_size : Int32,
) -> Int32 {
  moonbit_cuda_llm_mul_broadcast(left, right, output, n, right_size)
}

///|
#borrow(left, right, output)
extern "c" fn moonbit_cuda_llm_concat2(
  left : GpuBuffer,
  right : GpuBuffer,
  output : GpuBuffer,
  outer : Int32,
  left_width : Int32,
  right_width : Int32,
  inner : Int32,
) -> Int32 = "moonbit_cuda_llm_concat2"

///|
pub fn llm_concat2(
  left : GpuBuffer,
  right : GpuBuffer,
  output : GpuBuffer,
  outer : Int32,
  left_width : Int32,
  right_width : Int32,
  inner : Int32,
) -> Int32 {
  moonbit_cuda_llm_concat2(
    left, right, output, outer, left_width, right_width, inner,
  )
}

///|
#borrow(input, output, metadata)
extern "c" fn moonbit_cuda_llm_direct_layout(
  input : GpuBuffer,
  output : GpuBuffer,
  metadata : GpuBuffer,
  n : Int32,
) -> Int32 = "moonbit_cuda_llm_direct_layout"

///|
pub fn llm_direct_layout(
  input : GpuBuffer,
  output : GpuBuffer,
  metadata : GpuBuffer,
  n : Int32,
) -> Int32 {
  moonbit_cuda_llm_direct_layout(input, output, metadata, n)
}

///|
extern "c" fn moonbit_cuda_free(ptr : GpuBuffer) -> Int32 = "moonbit_cuda_free"

///|
pub fn free(ptr : GpuBuffer) -> Int32 {
  moonbit_cuda_free(ptr)
}
