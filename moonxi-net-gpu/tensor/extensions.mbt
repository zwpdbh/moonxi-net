// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
pub extend GpuTensor with @ts.Tensor::{
  broadcast_to,
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
  div_elem,
  reduce_sum_to,
  sub,
}

///|
pub extend GpuTensor with @ts.BlasTensor::{transpose, matmul}

///|
pub extend GpuTensor with @ts.ImageTensor::{
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
pub extend GpuTensor with @ts.ImageBackwardOps::{
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

///|
pub extend GpuTensor with @optimizer.BatchSgdMomentumAutoclip::{
  batch_sgd_momentum_autoclip,
}

///|
pub extend GpuTensor with @optimizer.BatchAdamAutoclip::{batch_adam_autoclip}

///|
pub extend GpuTensor with @optimizer.BatchRmspropAutoclip::{
  batch_rmsprop_autoclip,
}
