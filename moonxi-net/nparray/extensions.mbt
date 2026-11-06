// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
pub extend NpArray with @debug.Debug::{to_repr}

///|
pub extend NpArray with Add::{add}

///|
pub extend NpArray with Sub::{sub}

///|
pub extend NpArray with @tensor.Tensor::{
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
  dims,
  zeros_like,
  sqrt,
  size,
  from_host,
  div_elem,
  reduce_sum_to,
}

///|
pub extend NpArray with @tensor.BlasTensor::{transpose, matmul}

///|
pub extend NpArray with @tensor.ImageTensor::{
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
pub extend NpArray with @tensor.ImageBackwardOps::{
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
pub extend NpArray with @optimizer.BatchSgdMomentumAutoclip::{
  batch_sgd_momentum_autoclip,
}
