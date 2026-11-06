// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
fn llm_shape_size(shape : FixedArray[Int32]) -> Int32 {
  shape.iter().fold(init=1, (size, dim) => size * dim)
}

///|
fn llm_strides(shape : FixedArray[Int32]) -> FixedArray[Int32] {
  let strides = FixedArray::make(shape.length(), 1)
  for i in (shape.length() - 1)>..0 {
    strides[i] = strides[i + 1] * shape[i + 1]
  }
  strides
}

///|
fn llm_coords(
  index : Int32,
  shape : FixedArray[Int32],
  strides : FixedArray[Int32],
) -> FixedArray[Int32] {
  FixedArray::makei(shape.length(), dim => {
    if shape[dim] == 0 {
      0
    } else {
      index / strides[dim] % shape[dim]
    }
  })
}

///|
fn llm_dim(dim : Int32, rank : Int32) -> Int32 {
  let normalized = if dim < 0 { rank + dim } else { dim }
  if normalized < 0 || normalized >= rank {
    abort("dimension \{dim} is invalid for rank \{rank}")
  }
  normalized
}

///|
fn llm_reindex(
  input : NpArray,
  output_shape : Array[Int32],
  source_index : (FixedArray[Int32]) -> Int32,
) -> NpArray {
  let shape = FixedArray::from_array(output_shape)
  let strides = llm_strides(shape)
  let data = FixedArray::makei(llm_shape_size(shape), index => {
    input.data[source_index(llm_coords(index, shape, strides))]
  })
  { shape: { shape, }, data, }
}

///|
fn llm_index(coords : FixedArray[Int32], shape : FixedArray[Int32]) -> Int32 {
  let strides = llm_strides(shape)
  coords
  .iter()
  .zip(strides.iter())
  .fold(init=0, (index, pair) => index + pair.0 * pair.1)
}

///|
/// Reorder dimensions using a complete dimension permutation.
pub fn NpArray::permute(self : NpArray, axes : Array[Int32]) -> NpArray {
  let rank = self.shape.shape.length()
  if axes.length() != rank {
    abort("permute expected \{rank} axes, got \{axes.length()}")
  }
  let seen = FixedArray::make(rank, false)
  let normalized = axes.map(axis => {
    let axis = llm_dim(axis, rank)
    if seen[axis] {
      abort("permute contains duplicate axis \{axis}")
    }
    seen[axis] = true
    axis
  })
  let output_shape = normalized.map(axis => self.shape.shape[axis])
  let input_strides = llm_strides(self.shape.shape)
  llm_reindex(self, output_shape, output_coords => {
    let input_coords = FixedArray::make(rank, 0)
    for out_dim in 0..<rank {
      input_coords[normalized[out_dim]] = output_coords[out_dim]
    }
    input_coords
    .iter()
    .zip(input_strides.iter())
    .fold(init=0, (index, pair) => index + pair.0 * pair.1)
  })
}

///|
/// Insert a size-one dimension without copying tensor data.
pub fn NpArray::unsqueeze(self : NpArray, dim~ : Int32) -> NpArray {
  let rank = self.shape.shape.length()
  let dim = if dim < 0 { rank + dim + 1 } else { dim }
  if dim < 0 || dim > rank {
    abort("unsqueeze dimension is invalid for rank \{rank}")
  }
  let shape = Array::make(rank + 1, 0)
  for i in 0..<shape.length() {
    shape[i] = if i < dim {
      self.shape.shape[i]
    } else if i == dim {
      1
    } else {
      self.shape.shape[i - 1]
    }
  }
  self.view(FixedArray::from_array(shape))
}

///|
/// Copy a contiguous range along one dimension.
pub fn NpArray::narrow(
  self : NpArray,
  dim~ : Int32,
  start~ : Int32,
  length~ : Int32,
) -> NpArray {
  let rank = self.shape.shape.length()
  let dim = llm_dim(dim, rank)
  if start < 0 || length < 0 || start + length > self.shape.shape[dim] {
    abort("narrow range is out of bounds")
  }
  let output_shape = self.shape.shape.iter().collect()
  output_shape[dim] = length
  llm_reindex(self, output_shape, output_coords => {
    let input_coords = output_coords.copy()
    input_coords[dim] = input_coords[dim] + start
    llm_index(input_coords, self.shape.shape)
  })
}

///|
/// Select one index along a dimension and remove that dimension.
pub fn NpArray::select(self : NpArray, dim~ : Int32, index~ : Int32) -> NpArray {
  let dim = llm_dim(dim, self.shape.shape.length())
  let selected = self.narrow(dim~, start=index, length=1)
  let shape : Array[Int32] = []
  for i in 0..<selected.shape.shape.length() {
    if i != dim {
      shape.push(selected.shape.shape[i])
    }
  }
  selected.view(FixedArray::from_array(shape))
}

///|
/// Concatenate equally ranked tensors along one dimension.
pub fn NpArray::concat(tensors : Array[NpArray], dim~ : Int32) -> NpArray {
  if tensors.is_empty() {
    abort("concat requires at least one tensor")
  }
  let rank = tensors[0].shape.shape.length()
  let dim = llm_dim(dim, rank)
  let output_shape = tensors[0].shape.shape.iter().collect()
  output_shape[dim] = 0
  for tensor in tensors {
    if tensor.shape.shape.length() != rank {
      abort("concat rank mismatch")
    }
    for axis in 0..<rank {
      if axis != dim && tensor.shape.shape[axis] != tensors[0].shape.shape[axis] {
        abort("concat shape mismatch")
      }
    }
    output_shape[dim] = output_shape[dim] + tensor.shape.shape[dim]
  }
  let mut outer = 1
  let mut inner = 1
  for axis in 0..<dim {
    outer = outer * tensors[0].shape.shape[axis]
  }
  for axis in (dim + 1)..<rank {
    inner = inner * tensors[0].shape.shape[axis]
  }
  let data = FixedArray::make(
    llm_shape_size(FixedArray::from_array(output_shape)),
    0.0F,
  )
  let output_width = output_shape[dim]
  let mut concatenated_width = 0
  for tensor in tensors {
    let width = tensor.shape.shape[dim]
    for outer_index in 0..<outer {
      for column in 0..<width {
        for inner_index in 0..<inner {
          let source = (outer_index * width + column) * inner + inner_index
          let target = (
              outer_index * output_width + concatenated_width + column
            ) *
            inner +
            inner_index
          data[target] = tensor.data[source]
        }
      }
    }
    concatenated_width = concatenated_width + width
  }
  { shape: { shape: FixedArray::from_array(output_shape), }, data, }
}

///|
/// Broadcast a tensor to a compatible output shape.
pub fn NpArray::expand(self : NpArray, shape : Array[Int32]) -> NpArray {
  NpArray::zeros(FixedArray::from_array(shape)) + self
}

///|
/// Repeat each value along a dimension a fixed number of times.
pub fn NpArray::repeat_interleave(
  self : NpArray,
  repeats~ : Int32,
  dim~ : Int32,
) -> NpArray {
  let rank = self.shape.shape.length()
  let dim = llm_dim(dim, rank)
  if repeats < 0 {
    abort("repeat_interleave repeats must not be negative")
  }
  let output_shape = self.shape.shape.iter().collect()
  output_shape[dim] = output_shape[dim] * repeats
  llm_reindex(self, output_shape, output_coords => {
    let input_coords = output_coords.copy()
    input_coords[dim] = input_coords[dim] / repeats
    llm_index(input_coords, self.shape.shape)
  })
}

///|
/// Matrix multiplication with matching leading batch dimensions.
pub fn NpArray::batched_matmul(left : NpArray, right : NpArray) -> NpArray {
  let left_shape = left.shape.shape
  let right_shape = right.shape.shape
  if left_shape.length() < 2 || right_shape.length() < 2 {
    abort("batched_matmul requires tensors of at least rank 2")
  }
  let left_rank = left_shape.length()
  let right_rank = right_shape.length()
  if left_rank != right_rank {
    abort("batched_matmul requires matching ranks")
  }
  let m = left_shape[left_rank - 2]
  let k = left_shape[left_rank - 1]
  if right_shape[right_rank - 2] != k {
    abort("batched_matmul inner dimensions do not match")
  }
  let p = right_shape[right_rank - 1]
  let output_shape = left_shape.iter().collect()
  output_shape[left_rank - 1] = p
  for axis in 0..<(left_rank - 2) {
    if left_shape[axis] != right_shape[axis] {
      abort("batched_matmul batch dimensions do not match")
    }
  }
  let batch = llm_shape_size(
    FixedArray::from_array(left_shape.iter().take(left_rank - 2).collect()),
  )
  let data = FixedArray::make(batch * m * p, 0.0F)
  for batch_index in 0..<batch {
    for row in 0..<m {
      for inner in 0..<k {
        let left_value = left.data[batch_index * m * k + row * k + inner]
        for column in 0..<p {
          let output_index = batch_index * m * p + row * p + column
          data[output_index] = data[output_index] +
            left_value * right.data[batch_index * k * p + inner * p + column]
        }
      }
    }
  }
  { shape: { shape: FixedArray::from_array(output_shape), }, data, }
}

///|
/// Apply a rank-2 [out, in] weight to the final input dimension.
pub fn NpArray::linear(input : NpArray, weight : NpArray) -> NpArray {
  let input_shape = input.shape.shape
  let weight_shape = weight.shape.shape
  if input_shape.length() < 1 || weight_shape.length() != 2 {
    abort("linear expects input of rank >= 1 and a rank-2 weight")
  }
  let input_size = input_shape[input_shape.length() - 1]
  if weight_shape[1] != input_size {
    abort("linear input and weight dimensions do not match")
  }
  let rows = input.data.length() / input_size
  let output_size = weight_shape[0]
  let data = FixedArray::make(rows * output_size, 0.0F)
  for row in 0..<rows {
    for out in 0..<output_size {
      let mut sum = 0.0F
      for inner in 0..<input_size {
        sum = sum +
          input.data[row * input_size + inner] *
          weight.data[out * input_size + inner]
      }
      data[row * output_size + out] = sum
    }
  }
  let output_shape = input_shape.iter().collect()
  output_shape[input_shape.length() - 1] = output_size
  { shape: { shape: FixedArray::from_array(output_shape), }, data, }
}

///|
/// Apply a stable softmax along the final dimension.
pub fn NpArray::softmax_last(input : NpArray) -> NpArray {
  let rank = input.shape.shape.length()
  if rank == 0 {
    abort("softmax_last requires at least one dimension")
  }
  let width = input.shape.shape[rank - 1]
  let data = FixedArray::make(input.data.length(), 0.0F)
  let rows = if width == 0 { 0 } else { input.data.length() / width }
  for row in 0..<rows {
    let base = row * width
    let mut maximum = input.data[base]
    for column in 1..<width {
      maximum = Float::max(maximum, input.data[base + column])
    }
    let mut total = 0.0
    for column in 0..<width {
      let value = @math.exp((input.data[base + column] - maximum).to_double())
      data[base + column] = Float::from_double(value)
      total = total + value
    }
    for column in 0..<width {
      data[base + column] = Float::from_double(
        data[base + column].to_double() / total,
      )
    }
  }
  { shape: { shape: input.shape.shape.copy(), }, data, }
}

///|
/// Look up rows of a rank-2 embedding table using integer token IDs.
pub fn NpArray::embedding(
  weight : NpArray,
  token_ids : Array[Int32],
) -> NpArray {
  if weight.shape.shape.length() != 2 {
    abort("embedding weight must have rank 2")
  }
  let width = weight.shape.shape[1]
  let data = FixedArray::make(token_ids.length() * width, 0.0F)
  for row in 0..<token_ids.length() {
    let token_id = token_ids[row]
    if token_id < 0 || token_id >= weight.shape.shape[0] {
      abort("embedding token ID is out of bounds")
    }
    for column in 0..<width {
      data[row * width + column] = weight.data[token_id * width + column]
    }
  }
  { shape: @tensor.ShapeTensor::from_array([token_ids.length(), width]), data, }
}

///|
/// Apply the sigmoid-linear unit element by element.
pub fn NpArray::silu(input : NpArray) -> NpArray {
  {
    shape: { shape: input.shape.shape.copy(), },
    data: input.data.map(value => {
      Float::from_double(
        value.to_double() / (1.0 + @math.exp(-value.to_double())),
      )
    }),
  }
}

///|
/// Normalize the final dimension by root mean square and apply a weight.
pub fn NpArray::rms_norm(
  input : NpArray,
  weight : NpArray,
  eps~ : Float,
) -> NpArray {
  let rank = input.shape.shape.length()
  if rank == 0 || weight.shape.shape.length() != 1 {
    abort("rms_norm expects ranked input and a rank-1 weight")
  }
  let width = input.shape.shape[rank - 1]
  if width == 0 || weight.shape.shape[0] != width {
    abort("rms_norm weight dimension does not match input")
  }
  let data = FixedArray::make(input.data.length(), 0.0F)
  for row in 0..<(input.data.length() / width) {
    let base = row * width
    let mut mean_square = 0.0F
    for column in 0..<width {
      let value = input.data[base + column]
      mean_square = mean_square + value * value
    }
    mean_square = mean_square / Float::from_int(width)
    let scale = Float::sqrt(mean_square + eps)
    for column in 0..<width {
      data[base + column] = input.data[base + column] /
        scale *
        weight.data[column]
    }
  }
  { shape: { shape: input.shape.shape.copy(), }, data, }
}

///|
/// Replace values with a scalar wherever a broadcast mask is nonzero.
pub fn NpArray::masked_fill(
  input : NpArray,
  mask : NpArray,
  value~ : Float,
) -> NpArray {
  broadcast_op(
    (input_value, mask_value) => {
      if mask_value != 0.0F {
        value
      } else {
        input_value
      }
    },
    input,
    mask,
  )
}

///|
/// Return the final-dimension maximum index as an F32 tensor of IDs.
pub fn NpArray::argmax_last(input : NpArray) -> NpArray {
  let rank = input.shape.shape.length()
  if rank == 0 {
    abort("argmax_last requires at least one dimension")
  }
  let width = input.shape.shape[rank - 1]
  if width == 0 {
    abort("argmax_last does not accept an empty final dimension")
  }
  let output_shape = input.shape.shape.iter().take(rank - 1).collect()
  let rows = input.data.length() / width
  let data = FixedArray::make(rows, 0.0F)
  for row in 0..<rows {
    let base = row * width
    let mut selected = 0
    for column in 1..<width {
      if input.data[base + column] > input.data[base + selected] {
        selected = column
      }
    }
    data[row] = Float::from_int(selected)
  }
  { shape: @tensor.ShapeTensor::from_array(output_shape), data, }
}
