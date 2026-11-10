// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT
//
// moonxi-net minimal tutorial — build a trainable linear model from scratch
//
// Four core abstractions:
//   1. Tensor trait — tensor computation interface (extends Add + Sub, usable with operators)
//   2. NpArray      — CPU tensor implementation
//   3. Linear[T]    — generic model layer
//   4. Grad[T]      — tape-based automatic reverse-mode differentiation
//
// All tensors are [1,1] scalars, trained sample-by-sample, no broadcasting.

// =============================================================================
// 1. Tensor trait
// =============================================================================
//
// Extends Add + Sub, so types implementing Tensor can use + - operators directly.
// Self can be NpArray (CPU), GpuTensor (GPU), or even Grad[T] (automatic differentiation).

///|
trait Tensor: Add + Sub + Mul {
  fn dims(Self) -> FixedArray[Int32]
  fn zeros(dims : FixedArray[Int32]) -> Self
  fn scale(Self, Float) -> Self
  fn square(Self) -> Self
  fn mean(Self) -> Self
  fn scalar(Float) -> Self
  fn size(Self) -> Int32
}

// =============================================================================
// 2. NpArray — CPU tensor
// =============================================================================

///|
#derive(Debug)
struct NpArray {
  dims : FixedArray[Int32]
  data : FixedArray[Float]
}

///|
fn total_size(dims : FixedArray[Int32]) -> Int32 {
  dims.iter().fold(init=1, fn(acc, d) { acc * d })
}

///|
impl Tensor for NpArray with fn dims(x : NpArray) -> FixedArray[Int32] {
  x.dims
}

///|
impl Tensor for NpArray with fn zeros(dims : FixedArray[Int32]) -> NpArray {
  { dims, data: FixedArray::make(total_size(dims), (0.0 : Float)), }
}

///|
impl Tensor for NpArray with fn scale(x : NpArray, s : Float) -> NpArray {
  let n = x.data.length()
  { dims: x.dims, data: FixedArray::makei(n, fn(i) { x.data[i] * s }), }
}

///|
impl Mul for NpArray with fn mul(x : NpArray, y : NpArray) -> NpArray {
  let n = x.data.length()
  { dims: x.dims, data: FixedArray::makei(n, fn(i) { x.data[i] * y.data[i] }), }
}

///|
impl Tensor for NpArray with fn square(x : NpArray) -> NpArray {
  let n = x.data.length()
  { dims: x.dims, data: FixedArray::makei(n, fn(i) { x.data[i] * x.data[i] }), }
}

///|
impl Tensor for NpArray with fn mean(x : NpArray) -> NpArray {
  let mut sum = (0.0 : Float)
  for i in 0..<x.data.length() {
    sum = sum + x.data[i]
  }
  let n = Float::from_int(x.data.length())
  {
    dims: FixedArray::from_iter([1].iter()),
    data: FixedArray::make(1, sum / n),
  }
}

///|
impl Tensor for NpArray with fn scalar(v : Float) -> NpArray {
  { dims: FixedArray::from_iter([1].iter()), data: FixedArray::make(1, v), }
}

///|
impl Tensor for NpArray with fn size(x : NpArray) -> Int32 {
  x.data.length()
}

///|
impl Add for NpArray with fn add(x : NpArray, y : NpArray) -> NpArray {
  let n = x.data.length()
  { dims: x.dims, data: FixedArray::makei(n, fn(i) { x.data[i] + y.data[i] }), }
}

///|
impl Sub for NpArray with fn sub(x : NpArray, y : NpArray) -> NpArray {
  let n = x.data.length()
  { dims: x.dims, data: FixedArray::makei(n, fn(i) { x.data[i] - y.data[i] }), }
}

// =============================================================================
// 3. Linear[T] — generic model layer
// =============================================================================
//
// T = NpArray → forward pass only
// T = Grad[NpArray] → automatic gradient tracking (same forward function!)

///|
struct Linear[T] {
  w : T
  b : T
}

/// Forward computation: y = x * w + b

///|
fn[T : Tensor] Linear::forward(self : Linear[T], x : T) -> T {
  x * self.w + self.b
}

// =============================================================================
// 4. Grad[T] — automatic reverse-mode differentiation
// =============================================================================
//
// Grad[T] wraps tensor T, adding gradient tracking:
// - value: forward value
// - grad:  gradient (always present, initialized to zeros)
//
// Key: Grad[T] also implements Tensor (including Add + Sub),
// so Linear[Grad[NpArray]].forward automatically records backward closures on the tape.

///|
struct Grad[T] {
  mut value : T
  mut grad : T
}

///|
/// Global tape, stores backward closures
let tape : Array[() -> Unit] = Array(capacity=0)

///|
fn clear_tape() -> Unit {
  tape.clear()
}

/// Wrap a tensor as a Grad node, gradient initialized to zeros

///|
fn[T : Tensor] no_grad(t : T) -> Grad[T] {
  { value: t, grad: T::zeros(t.dims()), }
}

/// Record a backward closure on the tape. Gradient is accumulated directly onto input.grad.

///|
fn[T : Tensor] record(
  input : Grad[T],
  output : Grad[T],
  grad_fn : (T) -> T,
) -> Unit {
  tape.push(fn() -> Unit { input.grad = input.grad + grad_fn(output.grad) })
}

/// Backward pass: start from loss, set gradient to 1, replay tape in reverse order

///|
fn[T : Tensor] Grad::backward(self : Grad[T]) -> Unit {
  self.grad = T::scalar((1.0 : Float))
  for i in tape.length()>..0 {
    tape[i]()
  }
}

// --- Grad[T] implements Tensor ---
// Each operation: compute forward value + record backward closure

///|
impl[T : Tensor] Tensor for Grad[T] with fn dims(x : Grad[T]) -> FixedArray[
  Int32,
] {
  x.value.dims()
}

///|
impl[T : Tensor] Tensor for Grad[T] with fn zeros(dims : FixedArray[Int32]) -> Grad[
  T,
] {
  no_grad(T::zeros(dims))
}

///|
/// scale: z = x * s, ∂z/∂x = s
impl[T : Tensor] Tensor for Grad[T] with fn scale(x : Grad[T], s : Float) -> Grad[
  T,
] {
  let out : Grad[T] = {
    value: x.value.scale(s),
    grad: T::zeros(x.value.scale(s).dims()),
  }
  record(x, out, fn(g) { g.scale(s) })
  out
}

///|
/// mul: z = x * y, ∂z/∂x = y, ∂z/∂y = x
impl[T : Tensor] Mul for Grad[T] with fn mul(x : Grad[T], y : Grad[T]) -> Grad[
  T,
] {
  let saved_x = x.value
  let saved_y = y.value
  let out : Grad[T] = {
    value: x.value * y.value,
    grad: T::zeros((x.value * y.value).dims()),
  }
  record(x, out, fn(g) { g * saved_y })
  record(y, out, fn(g) { g * saved_x })
  out
}

///|
/// square: z = x², ∂z/∂x = 2x
impl[T : Tensor] Tensor for Grad[T] with fn square(x : Grad[T]) -> Grad[T] {
  let saved_x = x.value
  let out : Grad[T] = {
    value: x.value.square(),
    grad: T::zeros(x.value.square().dims()),
  }
  record(x, out, fn(g) { saved_x.scale((2.0 : Float)) * g })
  out
}

///|
/// mean: z = mean(x), ∂z/∂x_i = 1/n
impl[T : Tensor] Tensor for Grad[T] with fn mean(x : Grad[T]) -> Grad[T] {
  let n = x.value.size()
  let out : Grad[T] = {
    value: x.value.mean(),
    grad: T::zeros(x.value.mean().dims()),
  }
  record(x, out, fn(g) { g.scale((1.0 : Float) / Float::from_int(n)) })
  out
}

///|
impl[T : Tensor] Tensor for Grad[T] with fn scalar(v : Float) -> Grad[T] {
  no_grad(T::scalar(v))
}

///|
impl[T : Tensor] Tensor for Grad[T] with fn size(x : Grad[T]) -> Int32 {
  x.value.size()
}

// --- Grad[T] implements Add + Sub ---
// Forward uses operators, backward records gradient closures

///|
/// add: z = x + y, ∂z/∂x = 1, ∂z/∂y = 1
impl[T : Tensor] Add for Grad[T] with fn add(x : Grad[T], y : Grad[T]) -> Grad[
  T,
] {
  let out : Grad[T] = {
    value: x.value + y.value,
    grad: T::zeros((x.value + y.value).dims()),
  }
  record(x, out, fn(g) { g })
  record(y, out, fn(g) { g })
  out
}

///|
/// sub: z = x - y, ∂z/∂x = 1, ∂z/∂y = -1
impl[T : Tensor] Sub for Grad[T] with fn sub(x : Grad[T], y : Grad[T]) -> Grad[
  T,
] {
  let out : Grad[T] = {
    value: x.value - y.value,
    grad: T::zeros((x.value - y.value).dims()),
  }
  record(x, out, fn(g) { g })
  record(y, out, fn(g) { g.scale((-1.0 : Float)) })
  out
}

// =============================================================================
// 5. Training loop — learn y = 3x + 1
// =============================================================================
//
// All tensors are [1,1] scalars, sample-by-sample training, no broadcasting.

///|
fn main {
  // Training data: y = 3x + 1
  let xs : Array[Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
  let ys : Array[Float] = [4.0, 7.0, 10.0, 13.0, 16.0]

  // Initialize parameters w=0, b=0, wrapped in Grad for gradient tracking
  let w : Grad[NpArray] = no_grad(NpArray::scalar((0.0 : Float)))
  let b : Grad[NpArray] = no_grad(NpArray::scalar((0.0 : Float)))
  let model : Linear[Grad[NpArray]] = { w, b, }

  let lr : Float = (0.01 : Float)
  let params : Array[Grad[NpArray]] = [model.w, model.b]

  for epoch in 0..<=100 {
    for idx in 0..<xs.length() {
      clear_tape()
      // Zero gradients
      for p in params {
        p.grad = NpArray::zeros(p.value.dims())
      }

      // Input data
      let x = no_grad(NpArray::scalar(xs[idx]))
      let y = no_grad(NpArray::scalar(ys[idx]))

      // Forward: y_hat = x * w + b (+ and - are operators, automatically record gradients)
      let y_hat = model.forward(x)

      // loss = (y_hat - y)²
      let loss = (y_hat - y).square()

      // Backward pass
      loss.backward()

      // Gradient descent
      for p in params {
        p.value = p.value - p.grad.scale(lr)
      }
    }

    if epoch % 20 == 0 {
      println(
        "epoch \{epoch}: w=\{model.w.value.data[0]} b=\{model.b.value.data[0]}",
      )
    }
  }

  println("Learned: w=\{model.w.value.data[0]}, b=\{model.b.value.data[0]}")
  println("Expected: w=3.0, b=1.0")
}

///|
extend NpArray with Tensor::{scalar, zeros, dims, scale}

///|
extend Grad with Tensor::{square}
