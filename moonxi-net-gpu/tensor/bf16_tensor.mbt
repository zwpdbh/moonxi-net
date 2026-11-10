// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// Native CUDA tensor whose elements are stored as 16-bit bfloat values.
pub struct Bf16Tensor {
  data : @cuda.GpuBuffer
  shape : @ts.ShapeTensor
}

///|
pub fn Bf16Tensor::numel(self : Bf16Tensor) -> Int32 {
  @ts.ShapeTensor::size(self.shape)
}

///|
pub fn Bf16Tensor::ndim(self : Bf16Tensor) -> Int32 {
  @ts.ShapeTensor::ndim(self.shape)
}

///|
pub fn Bf16Tensor::dim(self : Bf16Tensor, axis : Int32) -> Int32 {
  @ts.ShapeTensor::dim(self.shape, axis)
}

///|
pub fn Bf16Tensor::alloc_shape(shape : Array[Int32]) -> Bf16Tensor {
  let count = shape.iter().fold(init=1, (size, dim) => size * dim)
  let bytes = Int64::from_int(count * 2)
  let data = if gpu_pool_state.use_pool {
    match gpu_pool_state.pool {
      Some(pool) => @cuda.pool_alloc(pool, bytes)
      None => @cuda.alloc(bytes)
    }
  } else {
    @cuda.alloc(bytes)
  }
  { data, shape: @ts.ShapeTensor::from_array(shape), }
}

///|
fn bf16_pack_value(value : Float) -> Int32 {
  let bits = value.reinterpret_as_uint()
  let rounded = bits + 0x7FFFU + ((bits >> 16) & 1U)
  (rounded >> 16).reinterpret_as_int()
}

///|
pub fn Bf16Tensor::from_host(
  data : FixedArray[Float],
  shape : Array[Int32],
) -> Bf16Tensor {
  let expected = shape.iter().fold(init=1, (size, dim) => size * dim)
  if data.length() != expected {
    abort("BF16 input data length does not match shape")
  }
  let packed = FixedArray::make((data.length() + 1) / 2, 0)
  for i in 0..<packed.length() {
    let low = bf16_pack_value(data[i * 2])
    let high = if i * 2 + 1 < data.length() {
      bf16_pack_value(data[i * 2 + 1])
    } else {
      0
    }
    packed[i] = low | (high << 16)
  }
  let buffer = @cuda.alloc(Int64::from_int(packed.length() * 4))
  let copy_data = packed.map(Float::reinterpret_from_int)
  check_cuda(@cuda.copy_h2d(buffer, copy_data), "copy BF16 values")
  { data: buffer, shape: @ts.ShapeTensor::from_array(shape), }
}

///|
pub fn Bf16Tensor::to_host(self : Bf16Tensor) -> FixedArray[Float] {
  let result = FixedArray::make(self.numel(), 0.0F)
  check_cuda(
    @cuda.bf16_to_host(self.data, result, self.numel()),
    "read BF16 values",
  )
  result
}

///|
pub fn Bf16Tensor::item(self : Bf16Tensor) -> Float {
  self.to_host()[0]
}

///|
pub fn Bf16Tensor::view(self : Bf16Tensor, shape : Array[Int32]) -> Bf16Tensor {
  let count = shape.iter().fold(init=1, (size, dim) => size * dim)
  if count != self.numel() {
    abort("BF16 view shape has a different element count")
  }
  { data: self.data, shape: @ts.ShapeTensor::from_array(shape), }
}

///|
fn bf16_dim(dim : Int32, rank : Int32) -> Int32 {
  let normalized = if dim < 0 { rank + dim } else { dim }
  if normalized < 0 || normalized >= rank {
    abort("BF16 dimension is invalid for rank \{rank}")
  }
  normalized
}

///|
fn bf16_direct_layout(
  input : Bf16Tensor,
  output_shape : Array[Int32],
  operation : Int32,
  dim : Int32,
  argument : Int32,
  axes : Array[Int32],
) -> Bf16Tensor {
  let rank = input.ndim()
  if rank > 4 || output_shape.length() != rank {
    abort("BF16 direct layout supports ranks 0 through 4")
  }
  let metadata = FixedArray::make(16, 1)
  metadata[0] = rank
  metadata[1] = operation
  metadata[2] = dim
  metadata[3] = argument
  for axis in 0..<rank {
    metadata[4 + axis] = input.dim(axis)
    metadata[8 + axis] = output_shape[axis]
  }
  for axis in 0..<axes.length() {
    metadata[12 + axis] = axes[axis]
  }
  let count = output_shape.iter().fold(init=1, (size, dim) => size * dim)
  let metadata_buffer = @cuda.alloc(64L)
  check_cuda(@cuda.raw_h2d(metadata_buffer, metadata), "copy BF16 metadata")
  let output = Bf16Tensor::alloc_shape(output_shape)
  check_cuda(
    @cuda.bf16_direct_layout(input.data, output.data, metadata_buffer, count),
    "BF16 direct layout",
  )
  check_cuda(@cuda.free(metadata_buffer), "free BF16 metadata")
  output
}

///|
pub fn Bf16Tensor::permute(
  self : Bf16Tensor,
  axes : Array[Int32],
) -> Bf16Tensor {
  let rank = self.ndim()
  if axes.length() != rank {
    abort("BF16 permute axis count does not match rank")
  }
  let seen = FixedArray::make(rank, false)
  let axes = axes.map(axis => {
    let axis = bf16_dim(axis, rank)
    if seen[axis] {
      abort("duplicate BF16 permute axis")
    }
    seen[axis] = true
    axis
  })
  bf16_direct_layout(self, axes.map(axis => self.dim(axis)), 0, 0, 0, axes)
}

///|
pub fn Bf16Tensor::unsqueeze(self : Bf16Tensor, dim~ : Int32) -> Bf16Tensor {
  let rank = self.ndim()
  let dim = if dim < 0 { rank + dim + 1 } else { dim }
  if dim < 0 || dim > rank {
    abort("BF16 unsqueeze dimension is invalid")
  }
  let shape = Array::make(rank + 1, 0)
  for i in 0..<shape.length() {
    shape[i] = if i < dim {
      self.dim(i)
    } else if i == dim {
      1
    } else {
      self.dim(i - 1)
    }
  }
  self.view(shape)
}

///|
pub fn Bf16Tensor::narrow(
  self : Bf16Tensor,
  dim~ : Int32,
  start~ : Int32,
  length~ : Int32,
) -> Bf16Tensor {
  let dim = bf16_dim(dim, self.ndim())
  if start < 0 || length < 0 || start + length > self.dim(dim) {
    abort("BF16 narrow range is out of bounds")
  }
  let shape = self.shape.shape.iter().collect()
  shape[dim] = length
  bf16_direct_layout(self, shape, 1, dim, start, [])
}

///|
pub fn Bf16Tensor::select(
  self : Bf16Tensor,
  dim~ : Int32,
  index~ : Int32,
) -> Bf16Tensor {
  let dim = bf16_dim(dim, self.ndim())
  let selected = self.narrow(dim~, start=index, length=1)
  let shape : Array[Int32] = []
  for i in 0..<selected.ndim() {
    if i != dim {
      shape.push(selected.dim(i))
    }
  }
  selected.view(shape)
}

///|
pub fn Bf16Tensor::concat(
  tensors : Array[Bf16Tensor],
  dim~ : Int32,
) -> Bf16Tensor {
  if tensors.length() != 2 {
    abort("BF16 concat currently requires exactly two tensors")
  }
  let left = tensors[0]
  let right = tensors[1]
  let rank = left.ndim()
  let dim = bf16_dim(dim, rank)
  if right.ndim() != rank {
    abort("BF16 concat rank mismatch")
  }
  for axis in 0..<rank {
    if axis != dim && left.dim(axis) != right.dim(axis) {
      abort("BF16 concat shape mismatch")
    }
  }
  let shape = left.shape.shape.iter().collect()
  shape[dim] = left.dim(dim) + right.dim(dim)
  let mut outer = 1
  let mut inner = 1
  for axis in 0..<dim {
    outer = outer * left.dim(axis)
  }
  for axis in (dim + 1)..<rank {
    inner = inner * left.dim(axis)
  }
  let output = Bf16Tensor::alloc_shape(shape)
  check_cuda(
    @cuda.bf16_concat2(
      left.data,
      right.data,
      output.data,
      outer,
      left.dim(dim),
      right.dim(dim),
      inner,
    ),
    "BF16 concat",
  )
  output
}

///|
pub fn Bf16Tensor::expand(
  self : Bf16Tensor,
  shape : Array[Int32],
) -> Bf16Tensor {
  if shape.length() != self.ndim() {
    abort("BF16 expand requires matching ranks")
  }
  for axis in 0..<shape.length() {
    if self.dim(axis) != 1 && self.dim(axis) != shape[axis] {
      abort("BF16 expand shape mismatch")
    }
  }
  bf16_direct_layout(self, shape, 2, 0, 0, [])
}

///|
pub fn Bf16Tensor::repeat_interleave(
  self : Bf16Tensor,
  repeats~ : Int32,
  dim~ : Int32,
) -> Bf16Tensor {
  let dim = bf16_dim(dim, self.ndim())
  if repeats <= 0 {
    abort("BF16 repeats must be positive")
  }
  let shape = self.shape.shape.iter().collect()
  shape[dim] = shape[dim] * repeats
  bf16_direct_layout(self, shape, 3, dim, repeats, [])
}

///|
pub fn Bf16Tensor::linear(
  input : Bf16Tensor,
  weight : Bf16Tensor,
) -> Bf16Tensor {
  if input.ndim() < 1 || weight.ndim() != 2 {
    abort("BF16 linear expects ranked input and a rank-2 weight")
  }
  let input_size = input.dim(input.ndim() - 1)
  if weight.dim(1) != input_size {
    abort("BF16 linear dimensions do not match")
  }
  let rows = input.numel() / input_size
  let output_size = weight.dim(0)
  let shape = input.shape.shape.iter().collect()
  shape[input.ndim() - 1] = output_size
  let output = Bf16Tensor::alloc_shape(shape)
  let context = @cuda.get_context(0)
  check_cuda(
    @cuda.bf16_linear(
      context.cublas,
      input.data,
      weight.data,
      output.data,
      rows,
      input_size,
      output_size,
    ),
    "BF16 linear",
  )
  output
}

///|
pub fn Bf16Tensor::batched_matmul(
  left : Bf16Tensor,
  right : Bf16Tensor,
) -> Bf16Tensor {
  if left.ndim() < 2 || left.ndim() != right.ndim() {
    abort("BF16 batched matmul requires matching ranks of at least 2")
  }
  let rank = left.ndim()
  let m = left.dim(rank - 2)
  let k = left.dim(rank - 1)
  let p = right.dim(rank - 1)
  if right.dim(rank - 2) != k {
    abort("BF16 batched matmul inner dimensions do not match")
  }
  let mut batch = 1
  for axis in 0..<(rank - 2) {
    if left.dim(axis) != right.dim(axis) {
      abort("BF16 batch dimensions do not match")
    }
    batch = batch * left.dim(axis)
  }
  let shape = left.shape.shape.iter().collect()
  shape[rank - 1] = p
  let output = Bf16Tensor::alloc_shape(shape)
  let context = @cuda.get_context(0)
  check_cuda(
    @cuda.bf16_batched_matmul(
      context.cublas,
      left.data,
      right.data,
      output.data,
      batch,
      m,
      k,
      p,
      0,
    ),
    "BF16 batched matmul",
  )
  output
}

///|
pub fn Bf16Tensor::softmax_last(input : Bf16Tensor) -> Bf16Tensor {
  if input.ndim() == 0 {
    abort("BF16 softmax requires ranked input")
  }
  let width = input.dim(input.ndim() - 1)
  let rows = if width == 0 { 0 } else { input.numel() / width }
  let output = Bf16Tensor::alloc_st(input.shape)
  check_cuda(
    @cuda.bf16_softmax(input.data, output.data, width, rows),
    "BF16 softmax",
  )
  output
}

///|
pub fn Bf16Tensor::alloc_st(shape : @ts.ShapeTensor) -> Bf16Tensor {
  Bf16Tensor::alloc_shape(shape.shape.iter().collect())
}

///|
pub fn Bf16Tensor::embedding(
  weight : Bf16Tensor,
  token_ids : Array[Int32],
) -> Bf16Tensor {
  if weight.ndim() != 2 {
    abort("BF16 embedding weight must have rank 2")
  }
  let width = weight.dim(1)
  let indices = FixedArray::makei(token_ids.length() * width, index => {
    let row = index / width
    let column = index % width
    token_ids[row] * width + column
  })
  let index_buffer = @cuda.alloc(Int64::from_int(indices.length() * 4))
  check_cuda(@cuda.raw_h2d(index_buffer, indices), "copy BF16 embedding IDs")
  let output = Bf16Tensor::alloc_shape([token_ids.length(), width])
  check_cuda(
    @cuda.bf16_gather(weight.data, index_buffer, output.data, indices.length()),
    "BF16 embedding",
  )
  check_cuda(@cuda.free(index_buffer), "free BF16 embedding IDs")
  output
}

///|
pub fn Bf16Tensor::silu(input : Bf16Tensor) -> Bf16Tensor {
  let output = Bf16Tensor::alloc_st(input.shape)
  check_cuda(
    @cuda.bf16_silu(input.data, output.data, input.numel()),
    "BF16 SiLU",
  )
  output
}

///|
pub fn Bf16Tensor::rms_norm(
  input : Bf16Tensor,
  weight : Bf16Tensor,
  eps~ : Float,
) -> Bf16Tensor {
  if input.ndim() == 0 || weight.ndim() != 1 {
    abort("BF16 RMSNorm expects ranked input and rank-1 weight")
  }
  let width = input.dim(input.ndim() - 1)
  if width == 0 || weight.dim(0) != width {
    abort("BF16 RMSNorm weight does not match input")
  }
  let rows = input.numel() / width
  let output = Bf16Tensor::alloc_st(input.shape)
  check_cuda(
    @cuda.bf16_rms_norm(input.data, weight.data, output.data, width, rows, eps),
    "BF16 RMSNorm",
  )
  output
}

///|
pub fn Bf16Tensor::masked_fill(
  input : Bf16Tensor,
  mask : Bf16Tensor,
  value~ : Float,
) -> Bf16Tensor {
  if input.numel() % mask.numel() != 0 {
    abort("BF16 mask is not broadcast-compatible")
  }
  let output = Bf16Tensor::alloc_st(input.shape)
  check_cuda(
    @cuda.bf16_masked_fill(
      input.data,
      mask.data,
      output.data,
      input.numel(),
      mask.numel(),
      value,
    ),
    "BF16 masked fill",
  )
  output
}

///|
pub fn Bf16Tensor::argmax_last(input : Bf16Tensor) -> Bf16Tensor {
  if input.ndim() == 0 {
    abort("BF16 argmax requires ranked input")
  }
  let width = input.dim(input.ndim() - 1)
  if width == 0 {
    abort("BF16 argmax does not accept an empty final dimension")
  }
  let rows = input.numel() / width
  let output = Bf16Tensor::alloc_shape(
    input.shape.shape.iter().take(input.ndim() - 1).collect(),
  )
  check_cuda(
    @cuda.bf16_argmax(input.data, output.data, rows, width),
    "BF16 argmax",
  )
  output
}

///|
pub fn Bf16Tensor::persistent(self : Bf16Tensor) -> Bf16Tensor {
  gpu_with_permanent_allocation(() => {
    bf16_direct_layout(self, self.shape.shape.iter().collect(), 4, 0, 0, [])
  })
}

///|
pub impl Add for Bf16Tensor with fn add(left, right) {
  if left.numel() != right.numel() {
    abort("BF16 addition requires matching element counts")
  }
  let output = Bf16Tensor::alloc_st(left.shape)
  check_cuda(
    @cuda.bf16_add(left.data, right.data, output.data, left.numel()),
    "BF16 add",
  )
  output
}

///|
pub extend Bf16Tensor with Add::{add}

///|
pub fn Bf16Tensor::mul_elem(
  left : Bf16Tensor,
  right : Bf16Tensor,
) -> Bf16Tensor {
  let output = Bf16Tensor::alloc_st(left.shape)
  if left.numel() % right.numel() != 0 {
    abort("BF16 multiplication operands are not broadcast-compatible")
  }
  check_cuda(
    @cuda.bf16_mul(
      left.data,
      right.data,
      output.data,
      left.numel(),
      right.numel(),
    ),
    "BF16 multiplication",
  )
  output
}

///|
pub fn Bf16Tensor::scale(input : Bf16Tensor, scalar : Float) -> Bf16Tensor {
  let output = Bf16Tensor::alloc_st(input.shape)
  check_cuda(
    @cuda.bf16_scale(input.data, output.data, input.numel(), scalar),
    "BF16 scale",
  )
  output
}
