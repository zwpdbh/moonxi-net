// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// Generic Conv2d with weight and bias.
pub(all) struct Conv2d[T] {
  mut weight : T
  mut bias : T
}

///|
/// Generic BatchNorm2d with gamma, beta, running_mean, running_var.
pub(all) struct BatchNorm2d[T] {
  mut gamma : T
  mut beta : T
  mut running_mean : T
  mut running_var : T
}

///|
/// Generic Linear layer.
pub(all) struct Linear[T] {
  mut w : T
  mut b : T
}

///|
/// Generic ResNet BasicBlock.
pub(all) struct BasicBlock[T] {
  mut conv1 : Conv2d[T]
  mut bn1 : BatchNorm2d[T]
  mut conv2 : Conv2d[T]
  mut bn2 : BatchNorm2d[T]
  mut shortcut : Conv2d[T]?
  mut shortcut_bn : BatchNorm2d[T]?
}

///|
/// Generic ResNet layer (sequence of BasicBlocks).
pub(all) struct ResNetLayer[T] {
  mut blocks : Array[BasicBlock[T]]
}

///|
/// Generic ResNet-18 model.
pub(all) struct ResNet18[T] {
  mut stem_conv : Conv2d[T]
  mut stem_bn : BatchNorm2d[T]
  mut layer1 : ResNetLayer[T]
  mut layer2 : ResNetLayer[T]
  mut layer3 : ResNetLayer[T]
  mut layer4 : ResNetLayer[T]
  mut head_weight : T
  mut head_bias : T
}

///|
/// Linear forward: y = x @ w + b
pub fn[T : @tensor.Tensor + @tensor.BlasTensor] Linear::forward(
  self : Linear[T],
  x : T,
) -> T {
  @tensor.Tensor::add(@tensor.BlasTensor::matmul(x, self.w), self.b)
}

///|
/// BasicBlock forward: conv1 → bn1 → relu → conv2 → bn2 → (+shortcut) → relu
pub fn[T : @tensor.Tensor + @tensor.ImageTensor] BasicBlock::forward(
  self : BasicBlock[T],
  input : T,
  is_training~ : Bool,
  stride : Int32,
) -> T {
  let conv1_out = @tensor.ImageTensor::conv2d(
    input,
    self.conv1.weight,
    self.conv1.bias,
    stride,
    1,
  )
  let bn1_out = if is_training {
    @tensor.ImageTensor::batchnorm_training(
      conv1_out,
      self.bn1.gamma,
      self.bn1.beta,
      self.bn1.running_mean,
      self.bn1.running_var,
      0.1F,
      0.00001F,
    ).0
  } else {
    @tensor.ImageTensor::batchnorm_inference(
      conv1_out,
      self.bn1.gamma,
      self.bn1.beta,
      self.bn1.running_mean,
      self.bn1.running_var,
      0.00001F,
    )
  }
  let relu1 = @tensor.ImageTensor::relu(bn1_out)
  let conv2_out = @tensor.ImageTensor::conv2d(
    relu1,
    self.conv2.weight,
    self.conv2.bias,
    1,
    1,
  )
  let bn2_out = if is_training {
    @tensor.ImageTensor::batchnorm_training(
      conv2_out,
      self.bn2.gamma,
      self.bn2.beta,
      self.bn2.running_mean,
      self.bn2.running_var,
      0.1F,
      0.00001F,
    ).0
  } else {
    @tensor.ImageTensor::batchnorm_inference(
      conv2_out,
      self.bn2.gamma,
      self.bn2.beta,
      self.bn2.running_mean,
      self.bn2.running_var,
      0.00001F,
    )
  }

  let shortcut : T = match self.shortcut {
    Some(sc) => {
      let sc_conv = @tensor.ImageTensor::conv2d(
        input,
        sc.weight,
        sc.bias,
        stride,
        0,
      )
      match self.shortcut_bn {
        Some(sc_bn) =>
          if is_training {
            @tensor.ImageTensor::batchnorm_training(
              sc_conv,
              sc_bn.gamma,
              sc_bn.beta,
              sc_bn.running_mean,
              sc_bn.running_var,
              0.1F,
              0.00001F,
            ).0
          } else {
            @tensor.ImageTensor::batchnorm_inference(
              sc_conv,
              sc_bn.gamma,
              sc_bn.beta,
              sc_bn.running_mean,
              sc_bn.running_var,
              0.00001F,
            )
          }
        None => sc_conv
      }
    }
    None => input
  }
  @tensor.ImageTensor::relu(@tensor.Tensor::add(bn2_out, shortcut))
}

///|
/// ResNetLayer forward (sequence of BasicBlocks).
/// first_stride is the stride for the first block only; rest use stride=1.
pub fn[T : @tensor.Tensor + @tensor.ImageTensor] ResNetLayer::forward(
  self : ResNetLayer[T],
  input : T,
  is_training~ : Bool,
  first_stride : Int32,
) -> T {
  let mut x = input
  for i in 0..<self.blocks.length() {
    let stride = if i == 0 { first_stride } else { 1 }
    x = self.blocks[i].forward(x, is_training~, stride)
  }
  x
}

///|
/// ResNet-18 feature extractor: input [batch, 3, H, W] → features [batch, 512]
/// Runs stem + 4 residual layers + adaptive avg pool + flatten.
pub fn[T : @tensor.Tensor + @tensor.ImageTensor] ResNet18::forward_features(
  self : ResNet18[T],
  input : T,
  is_training~ : Bool,
) -> T {
  let x = @tensor.ImageTensor::conv2d(
    input,
    self.stem_conv.weight,
    self.stem_conv.bias,
    2,
    3,
  )
  let x = if is_training {
    @tensor.ImageTensor::batchnorm_training(
      x,
      self.stem_bn.gamma,
      self.stem_bn.beta,
      self.stem_bn.running_mean,
      self.stem_bn.running_var,
      0.1F,
      0.00001F,
    ).0
  } else {
    @tensor.ImageTensor::batchnorm_inference(
      x,
      self.stem_bn.gamma,
      self.stem_bn.beta,
      self.stem_bn.running_mean,
      self.stem_bn.running_var,
      0.00001F,
    )
  }
  let x = @tensor.ImageTensor::relu(x)
  let x = @tensor.ImageTensor::maxpool2d(x, 3, 2)
  let x = self.layer1.forward(x, is_training~, 1)
  let x = self.layer2.forward(x, is_training~, 2)
  let x = self.layer3.forward(x, is_training~, 2)
  let x = self.layer4.forward(x, is_training~, 2)
  let x = @tensor.ImageTensor::adaptive_avg_pool2d(x, 1)
  let n = @tensor.Tensor::dims(x)[0]
  @tensor.Tensor::view(
    x,
    FixedArray::makei(2, i => if i == 0 { n } else { 512 }),
  )
}

///|
/// ResNet-18 forward: input [batch, 3, H, W] → output [batch, num_classes]
pub fn[T : @tensor.Tensor + @tensor.BlasTensor + @tensor.ImageTensor] ResNet18::forward(
  self : ResNet18[T],
  input : T,
  is_training~ : Bool,
) -> T {
  let x = self.forward_features(input, is_training~)
  @tensor.Tensor::add(
    @tensor.BlasTensor::matmul(
      x,
      @tensor.BlasTensor::transpose(self.head_weight),
    ),
    self.head_bias,
  )
}

///|
/// Collect all trainable parameters from ResNet-18.
/// When T = @grad.Grad[GpuTensor], fields are already Grad-wrapped — return directly.
/// Does NOT include running_mean/running_var.
pub fn[T] ResNet18::params(self : ResNet18[T]) -> Array[T] {
  let result : Array[T] = []

  // Stem conv
  result.push(self.stem_conv.weight)
  result.push(self.stem_conv.bias)
  // Stem bn (gamma, beta only - no running stats)
  result.push(self.stem_bn.gamma)
  result.push(self.stem_bn.beta)

  // Layers 1-4
  for layer in [self.layer1, self.layer2, self.layer3, self.layer4] {
    for block in layer.blocks {
      // conv1
      result.push(block.conv1.weight)
      result.push(block.conv1.bias)
      // bn1
      result.push(block.bn1.gamma)
      result.push(block.bn1.beta)
      // conv2
      result.push(block.conv2.weight)
      result.push(block.conv2.bias)
      // bn2
      result.push(block.bn2.gamma)
      result.push(block.bn2.beta)
      // shortcut
      match block.shortcut {
        Some(sc) => {
          result.push(sc.weight)
          result.push(sc.bias)
        }
        None => ()
      }
      // shortcut_bn
      match block.shortcut_bn {
        Some(sc_bn) => {
          result.push(sc_bn.gamma)
          result.push(sc_bn.beta)
        }
        None => ()
      }
    }
  }

  // Head
  result.push(self.head_weight)
  result.push(self.head_bias)

  result
}
