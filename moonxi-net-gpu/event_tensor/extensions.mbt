// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
pub extend CudaEventTensor with @tensor.Tensor::{
  zeros,
  broadcast_to,
  view,
  add_into,
  scale,
  scalar,
  mean,
  square,
  mul_elem,
  value,
  add,
  dims,
  zeros_like,
  sqrt,
  size,
  from_host,
  div_elem,
  reduce_sum_to,
  sub,
}

///|
pub extend CudaEventTensor with @tensor.BlasTensor::{transpose, matmul}

///|
pub extend CudaEventTensor with @tensor.ImageTensor::{
  conv2d,
  relu,
  cross_entropy_with_labels,
  softmax_cross_entropy,
  adaptive_avg_pool2d,
  maxpool2d,
  batchnorm_inference,
  batchnorm_training,
}

///|
pub extend CudaEventTensor with @tensor.ImageBackwardOps::{
  relu_backward,
  conv2d_backward_bias,
  conv2d_backward_data,
  softmax_ce_backward_labels,
  softmax_ce_backward,
  batchnorm_backward,
  conv2d_backward_weight,
  adaptive_avg_pool2d_backward,
  maxpool2d_backward,
}
