// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
fn[T : @tensor.Tensor + @tensor.BlasTensor] train_linear() -> Array[T] {
  // Model: y = w * x + b
  let w = @grad.no_grad(@tensor.Tensor::zeros([1, 1]))
  let b = @grad.no_grad(@tensor.Tensor::zeros([1, 1]))

  // Training data: y = 3x + 1
  let x = @grad.no_grad(
    @tensor.Tensor::from_host(
      FixedArray::makei(5, i => Float::from_int(i + 1)),
      [5, 1],
    ),
  )
  let y = @grad.no_grad(
    @tensor.Tensor::from_host(
      FixedArray::makei(5, i => Float::from_int((i + 1) * 3 + 1)),
      [5, 1],
    ),
  )

  let params : Array[@grad.Grad[T]] = [w, b]

  for _epoch in 0..<=500 {
    @grad.clear_tape()
    for p in params {
      p.grad = Some(None)
    }
    let y_hat = @tensor.Tensor::add(@tensor.BlasTensor::matmul(x, w), b)
    let loss = @tensor.Tensor::mean(
      @tensor.Tensor::square(@tensor.Tensor::sub(y_hat, y)),
    )
    loss.backward()
    for p in params {
      match p.grad {
        Some(Some(g)) =>
          p.value = @tensor.Tensor::sub(
            p.value,
            @tensor.Tensor::scale(g, 0.001),
          )
        _ => ()
      }
    }
  }

  params.map(g => g.value)
}

///|
fn main {
  @cuda.gpu_init(0)
  let params : Array[@gt.GpuTensor] = train_linear()
  let wv = params[0].value()
  let bv = params[1].value()
  println("Learned: w=\{wv.data[0]}, b=\{bv.data[0]}")
}
