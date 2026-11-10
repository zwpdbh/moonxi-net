// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
fn gpu_llm_shape_size(shape : Array[Int32]) -> Int32 {
  shape.iter().fold(init=1, (size, dim) => size * dim)
}

///|
fn gpu_llm_strides(shape : FixedArray[Int32]) -> FixedArray[Int32] {
  let strides = FixedArray::make(shape.length(), 1)
  for i in (shape.length() - 1)>..0 {
    strides[i] = strides[i + 1] * shape[i + 1]
  }
  strides
}

///|
fn gpu_llm_dim(dim : Int32, rank : Int32) -> Int32 {
  let normalized = if dim < 0 { rank + dim } else { dim }
  if normalized < 0 || normalized >= rank {
    abort("dimension is invalid for rank \{rank}")
  }
  normalized
}

///|
fn gpu_llm_coords(
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
/// Compute a flat row-major index for the reference gather implementation.
pub fn gpu_llm_index(
  coords : FixedArray[Int32],
  shape : FixedArray[Int32],
) -> Int32 {
  let strides = gpu_llm_strides(shape)
  coords
  .iter()
  .zip(strides.iter())
  .fold(init=0, (index, pair) => index + pair.0 * pair.1)
}

///|
/// Reference gather implementation using a host-generated index map.
pub fn gpu_llm_gather(
  input : GpuTensor,
  output_shape : Array[Int32],
  source_index : (FixedArray[Int32]) -> Int32,
) -> GpuTensor {
  let profiler = llm_profiler()
  profiler.host("gather_total_ms", () => {
    let count = gpu_llm_shape_size(output_shape)
    profiler.count("gather_calls", 1L)
    profiler.count("gather_index_elements", Int64::from_int(count))
    let shape = FixedArray::from_array(output_shape)
    let strides = gpu_llm_strides(shape)
    let indices = profiler.host("gather_index_build_ms", () => {
      FixedArray::makei(count, index => {
        source_index(gpu_llm_coords(index, shape, strides))
      })
    })
    let index_buffer = profiler.host("gather_index_alloc_ms", () => {
      profiler.count("gather_index_allocations", 1L)
      @cuda.alloc(Int64::from_int(count * 4))
    })
    profiler.host("gather_index_h2d_ms", () => {
      profiler.count("gather_index_h2d_bytes", Int64::from_int(count * 4))
      check_cuda(@cuda.raw_h2d(index_buffer, indices), "copy LLM indices")
    })
    let output = GpuTensor::alloc_shape(output_shape)
    profiler.cuda("gather_kernel_cuda_ms", () => {
      check_cuda(
        @cuda.llm_gather(input.data, index_buffer, output.data, count),
        "LLM gather",
      )
    })
    profiler.host("gather_index_free_ms", () => {
      profiler.count("gather_index_frees", 1L)
      check_cuda(@cuda.free(index_buffer), "free LLM indices")
    })
    output
  })
}

///|
fn gpu_llm_direct_layout(
  input : GpuTensor,
  output_shape : Array[Int32],
  operation : Int32,
  dim : Int32,
  argument : Int32,
  axes : Array[Int32],
) -> GpuTensor {
  let rank = input.ndim()
  if rank > 4 || output_shape.length() != rank {
    abort("direct layout currently supports ranks 0 through 4")
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
  let count = gpu_llm_shape_size(output_shape)
  let profiler = llm_profiler()
  profiler.count("direct_layout_calls", 1L)
  profiler.count("direct_layout_output_elements", Int64::from_int(count))
  let metadata_buffer = @cuda.alloc(64L)
  check_cuda(@cuda.raw_h2d(metadata_buffer, metadata), "copy layout metadata")
  let output = GpuTensor::alloc_shape(output_shape)
  profiler.cuda("direct_layout_cuda_ms", () => {
    check_cuda(
      @cuda.llm_direct_layout(input.data, output.data, metadata_buffer, count),
      "direct layout",
    )
  })
  check_cuda(@cuda.free(metadata_buffer), "free layout metadata")
  output
}

///|
/// Copy a tensor into storage that survives temporary workspace resets.
pub fn GpuTensor::persistent(self : GpuTensor) -> GpuTensor {
  gpu_with_permanent_allocation(() => {
    gpu_llm_direct_layout(self, self.shape.shape.iter().collect(), 4, 0, 0, [])
  })
}

///|
pub fn GpuTensor::permute(self : GpuTensor, axes : Array[Int32]) -> GpuTensor {
  let rank = self.ndim()
  if axes.length() != rank {
    abort("permute axis count does not match rank")
  }
  let seen = FixedArray::make(rank, false)
  let axes = axes.map(axis => {
    let axis = gpu_llm_dim(axis, rank)
    if seen[axis] {
      abort("duplicate permute axis")
    }
    seen[axis] = true
    axis
  })
  let output_shape = axes.map(axis => self.dim(axis))
  gpu_llm_direct_layout(self, output_shape, 0, 0, 0, axes)
}

///|
pub fn GpuTensor::unsqueeze(self : GpuTensor, dim~ : Int32) -> GpuTensor {
  let rank = self.ndim()
  let dim = if dim < 0 { rank + dim + 1 } else { dim }
  if dim < 0 || dim > rank {
    abort("unsqueeze dimension is invalid")
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
pub fn GpuTensor::narrow(
  self : GpuTensor,
  dim~ : Int32,
  start~ : Int32,
  length~ : Int32,
) -> GpuTensor {
  let dim = gpu_llm_dim(dim, self.ndim())
  if start < 0 || length < 0 || start + length > self.dim(dim) {
    abort("narrow range is out of bounds")
  }
  let output_shape = self.shape.shape.iter().collect()
  output_shape[dim] = length
  gpu_llm_direct_layout(self, output_shape, 1, dim, start, [])
}

///|
pub fn GpuTensor::select(
  self : GpuTensor,
  dim~ : Int32,
  index~ : Int32,
) -> GpuTensor {
  let dim = gpu_llm_dim(dim, self.ndim())
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
pub fn GpuTensor::concat(tensors : Array[GpuTensor], dim~ : Int32) -> GpuTensor {
  if tensors.length() != 2 {
    abort("GPU concat currently requires exactly two tensors")
  }
  let left = tensors[0]
  let right = tensors[1]
  let rank = left.ndim()
  let dim = gpu_llm_dim(dim, rank)
  if right.ndim() != rank {
    abort("concat rank mismatch")
  }
  let output_shape = left.shape.shape.iter().collect()
  output_shape[dim] = left.dim(dim) + right.dim(dim)
  for axis in 0..<rank {
    if axis != dim && right.dim(axis) != left.dim(axis) {
      abort("concat shape mismatch")
    }
  }
  let mut outer = 1
  let mut inner = 1
  for axis in 0..<dim {
    outer = outer * left.dim(axis)
  }
  for axis in (dim + 1)..<rank {
    inner = inner * left.dim(axis)
  }
  let output = GpuTensor::alloc_shape(output_shape)
  llm_profiler().count("concat_calls", 1L)
  llm_profiler().cuda("concat_cuda_ms", () => {
    check_cuda(
      @cuda.llm_concat2(
        left.data,
        right.data,
        output.data,
        outer,
        left.dim(dim),
        right.dim(dim),
        inner,
      ),
      "LLM concat",
    )
  })
  output
}

///|
pub fn GpuTensor::expand(self : GpuTensor, shape : Array[Int32]) -> GpuTensor {
  if shape.length() != self.ndim() {
    abort("expand currently requires matching ranks")
  }
  for axis in 0..<shape.length() {
    if self.dim(axis) != 1 && self.dim(axis) != shape[axis] {
      abort("expand shape mismatch")
    }
  }
  gpu_llm_direct_layout(self, shape, 2, 0, 0, [])
}

///|
pub fn GpuTensor::repeat_interleave(
  self : GpuTensor,
  repeats~ : Int32,
  dim~ : Int32,
) -> GpuTensor {
  let dim = gpu_llm_dim(dim, self.ndim())
  if repeats <= 0 {
    abort("repeat_interleave repeats must be positive")
  }
  let output_shape = self.shape.shape.iter().collect()
  output_shape[dim] = output_shape[dim] * repeats
  gpu_llm_direct_layout(self, output_shape, 3, dim, repeats, [])
}

///|
pub fn GpuTensor::linear(input : GpuTensor, weight : GpuTensor) -> GpuTensor {
  if input.ndim() < 1 || weight.ndim() != 2 {
    abort("linear expects ranked input and a rank-2 weight")
  }
  let input_size = input.dim(input.ndim() - 1)
  if weight.dim(1) != input_size {
    abort("linear input and weight dimensions do not match")
  }
  let rows = input.numel() / input_size
  let output_size = weight.dim(0)
  let output_shape = input.shape.shape.iter().collect()
  output_shape[input.ndim() - 1] = output_size
  let output = GpuTensor::alloc_shape(output_shape)
  let context = @cuda.get_context(0)
  llm_profiler().count("linear_calls", 1L)
  llm_profiler().cuda("linear_cuda_ms", () => {
    check_cuda(
      @cuda.matmul(
        context.cublas,
        input.data,
        weight.data,
        output.data,
        rows,
        input_size,
        output_size,
        false,
        true,
        1.0F,
        0.0F,
      ),
      "LLM linear",
    )
  })
  output
}

///|
pub fn GpuTensor::batched_matmul(
  left : GpuTensor,
  right : GpuTensor,
) -> GpuTensor {
  if left.ndim() < 2 || left.ndim() != right.ndim() {
    abort("batched_matmul requires matching ranks of at least 2")
  }
  let rank = left.ndim()
  let m = left.dim(rank - 2)
  let k = left.dim(rank - 1)
  let p = right.dim(rank - 1)
  if right.dim(rank - 2) != k {
    abort("batched_matmul inner dimensions do not match")
  }
  let mut batch = 1
  for axis in 0..<(rank - 2) {
    if left.dim(axis) != right.dim(axis) {
      abort("batched_matmul batch dimensions do not match")
    }
    batch = batch * left.dim(axis)
  }
  let output_shape = left.shape.shape.iter().collect()
  output_shape[rank - 1] = p
  let output = GpuTensor::alloc_shape(output_shape)
  let context = @cuda.get_context(0)
  llm_profiler().count("batched_matmul_calls", 1L)
  llm_profiler().cuda("batched_matmul_cuda_ms", () => {
    check_cuda(
      @cuda.llm_batched_matmul(
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
      "LLM batched matmul",
    )
  })
  output
}

///|
pub fn GpuTensor::softmax_last(input : GpuTensor) -> GpuTensor {
  if input.ndim() == 0 {
    abort("softmax_last requires ranked input")
  }
  let width = input.dim(input.ndim() - 1)
  let rows = if width == 0 { 0 } else { input.numel() / width }
  let output = GpuTensor::alloc_st(input.shape)
  llm_profiler().count("softmax_calls", 1L)
  llm_profiler().cuda("softmax_cuda_ms", () => {
    check_cuda(
      @cuda.llm_softmax(input.data, output.data, width, rows),
      "LLM softmax",
    )
  })
  output
}

///|
pub fn GpuTensor::embedding(
  weight : GpuTensor,
  token_ids : Array[Int32],
) -> GpuTensor {
  if weight.ndim() != 2 {
    abort("embedding weight must have rank 2")
  }
  let width = weight.dim(1)
  llm_profiler().count("embedding_calls", 1L)
  llm_profiler().count(
    "embedding_index_h2d_bytes",
    Int64::from_int(token_ids.length() * width * 4),
  )
  let indices = FixedArray::makei(token_ids.length() * width, index => {
    let row = index / width
    let column = index % width
    token_ids[row] * width + column
  })
  let index_buffer = @cuda.alloc(Int64::from_int(indices.length() * 4))
  check_cuda(@cuda.raw_h2d(index_buffer, indices), "copy embedding IDs")
  let output = GpuTensor::alloc_shape([token_ids.length(), width])
  llm_profiler().cuda("embedding_cuda_ms", () => {
    check_cuda(
      @cuda.llm_gather(weight.data, index_buffer, output.data, indices.length()),
      "embedding",
    )
  })
  check_cuda(@cuda.free(index_buffer), "free embedding IDs")
  output
}

///|
pub fn GpuTensor::silu(input : GpuTensor) -> GpuTensor {
  let output = GpuTensor::alloc_st(input.shape)
  llm_profiler().count("silu_calls", 1L)
  llm_profiler().cuda("silu_cuda_ms", () => {
    check_cuda(
      @cuda.llm_silu(input.data, output.data, input.numel()),
      "LLM SiLU",
    )
  })
  output
}

///|
pub fn GpuTensor::rms_norm(
  input : GpuTensor,
  weight : GpuTensor,
  eps~ : Float,
) -> GpuTensor {
  if input.ndim() == 0 || weight.ndim() != 1 {
    abort("rms_norm expects ranked input and rank-1 weight")
  }
  let width = input.dim(input.ndim() - 1)
  if width == 0 || weight.dim(0) != width {
    abort("rms_norm weight dimension does not match input")
  }
  let rows = input.numel() / width
  let output = GpuTensor::alloc_st(input.shape)
  llm_profiler().count("rms_norm_calls", 1L)
  llm_profiler().cuda("rms_norm_cuda_ms", () => {
    check_cuda(
      @cuda.llm_rms_norm(input.data, weight.data, output.data, width, rows, eps),
      "LLM RMSNorm",
    )
  })
  output
}

///|
pub fn GpuTensor::masked_fill(
  input : GpuTensor,
  mask : GpuTensor,
  value~ : Float,
) -> GpuTensor {
  if input.numel() % mask.numel() != 0 {
    abort("masked_fill mask is not broadcast-compatible")
  }
  let output = GpuTensor::alloc_st(input.shape)
  llm_profiler().count("masked_fill_calls", 1L)
  llm_profiler().cuda("masked_fill_cuda_ms", () => {
    check_cuda(
      @cuda.llm_masked_fill(
        input.data,
        mask.data,
        output.data,
        input.numel(),
        mask.numel(),
        value,
      ),
      "LLM masked fill",
    )
  })
  output
}

///|
pub fn GpuTensor::argmax_last(input : GpuTensor) -> GpuTensor {
  if input.ndim() == 0 {
    abort("argmax_last requires ranked input")
  }
  let width = input.dim(input.ndim() - 1)
  if width == 0 {
    abort("argmax_last does not accept an empty final dimension")
  }
  let rows = input.numel() / width
  let output_shape = input.shape.shape.iter().take(input.ndim() - 1).collect()
  let output = GpuTensor::alloc_shape(output_shape)
  llm_profiler().count("argmax_calls", 1L)
  llm_profiler().cuda("argmax_cuda_ms", () => {
    check_cuda(@cuda.argmax(input.data, output.data, rows, width), "LLM argmax")
  })
  output
}

///|
pub fn gpu_llm_mul_elem(left : GpuTensor, right : GpuTensor) -> GpuTensor {
  let output = GpuTensor::alloc_st(left.shape)
  llm_profiler().count("mul_elem_calls", 1L)
  llm_profiler().cuda("mul_elem_cuda_ms", () => {
    if left.numel() == right.numel() {
      check_cuda(
        @cuda.mul_elem(left.data, right.data, output.data, left.numel()),
        "LLM elementwise multiplication",
      )
    } else {
      if left.numel() % right.numel() != 0 {
        abort("multiplication operands are not broadcast-compatible")
      }
      check_cuda(
        @cuda.llm_mul_broadcast(
          left.data,
          right.data,
          output.data,
          left.numel(),
          right.numel(),
        ),
        "LLM broadcast multiplication",
      )
    }
  })
  output
}
