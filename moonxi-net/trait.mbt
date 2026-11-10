// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
#derive(Debug)
pub(all) struct TensorData {
  dims : FixedArray[Int32]
  data : FixedArray[Float]
}

///|
pub(open) trait Tensor {
  fn dims(Self) -> FixedArray[Int32]
  fn zeros(dims : FixedArray[Int32]) -> Self
  fn zeros_like(Self) -> Self
  fn from_host(data : FixedArray[Float], shape : Array[Int32]) -> Self
  fn square(Self) -> Self
  fn sqrt(Self) -> Self
  fn mean(Self) -> Self
  fn scale(Self, Float) -> Self
  fn mul_elem(Self, Self) -> Self
  fn div_elem(Self, Self) -> Self
  fn scalar(Float) -> Self
  fn size(Self) -> Int32
  fn reduce_sum_to(Self, target_dims : FixedArray[Int32]) -> Self
  fn value(Self) -> TensorData
  fn view(Self, new_shape : FixedArray[Int32]) -> Self
  fn add(Self, Self) -> Self
  fn add_into(Self, Self) -> Unit
  fn sub(Self, Self) -> Self
  /// Broadcast a scalar tensor to the given shape (fills target shape with the scalar value).
  fn broadcast_to(Self, target_shape : FixedArray[Int32]) -> Self
}

///|
pub(open) trait BlasTensor {
  fn matmul(Self, Self) -> Self
  fn transpose(Self) -> Self
}

///|
pub(open) trait ImageTensor {
  // ── Forward ──
  fn conv2d(Self, weight : Self, bias : Self, stride : Int32, padding : Int32) -> Self
  fn relu(Self) -> Self
  fn maxpool2d(Self, kernel_size : Int32, stride : Int32) -> Self
  fn adaptive_avg_pool2d(Self, output_size : Int32) -> Self
  fn batchnorm_training(
    Self,
    gamma : Self,
    beta : Self,
    running_mean : Self,
    running_var : Self,
    momentum : Float,
    eps : Float,
  ) -> (Self, Self, Self)
  fn batchnorm_inference(
    Self,
    gamma : Self,
    beta : Self,
    running_mean : Self,
    running_var : Self,
    eps : Float,
  ) -> Self
  fn softmax_cross_entropy(Self, targets : Self) -> Self
  /// Cross-entropy loss with integer class labels (stored as float tensor of shape [batch]).
  /// Equivalent to softmax_cross_entropy but avoids one-hot encoding.
  /// labels: tensor of shape [batch], each element is an integer class index.
  /// num_classes: number of classes (C), used by the kernel for logits shape [batch, C].
  fn cross_entropy_with_labels(Self, labels : Self, num_classes : Int32) -> Self
}

///|
pub(open) trait ImageBackwardOps {
  // ── Backward ──
  /// d(conv2d)/d(input)
  fn conv2d_backward_data(
    grad_output : Self,
    weight : Self,
    input : Self,
    stride : Int32,
    padding : Int32,
  ) -> Self

  /// d(conv2d)/d(weight)
  fn conv2d_backward_weight(
    grad_output : Self,
    input : Self,
    weight : Self,
    stride : Int32,
    padding : Int32,
  ) -> Self

  /// d(conv2d)/d(bias) — reduce_sum over spatial dims
  fn conv2d_backward_bias(grad_output : Self) -> Self

  /// d(relu)/d(input)
  fn relu_backward(grad_output : Self, input : Self) -> Self

  /// d(batchnorm)/d(input), d(gamma), d(beta) — computed simultaneously
  fn batchnorm_backward(
    grad_output : Self,
    input : Self,
    gamma : Self,
    save_mean : Self,
    save_inv_var : Self,
    eps : Float,
  ) -> (Self, Self, Self)

  /// d(maxpool2d)/d(input)
  fn maxpool2d_backward(
    grad_output : Self,
    input : Self,
    kernel_size : Int32,
    stride : Int32,
  ) -> Self

  /// d(adaptive_avg_pool2d)/d(input)
  fn adaptive_avg_pool2d_backward(grad_output : Self, input : Self) -> Self

  /// d(softmax_cross_entropy)/d(logits) = (softmax(logits) - targets) / batch
  /// The result should be scaled by grad_output externally if needed.
  fn softmax_ce_backward(logits : Self, targets : Self) -> Self
  /// Backward for cross_entropy_with_labels.
  /// d(logits[b][j]) = (softmax[j] - indicator(j == class_idx)) / batch
  fn softmax_ce_backward_labels(
    logits : Self,
    labels : Self,
    num_classes : Int32,
  ) -> Self
}
