#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using BF16 = __nv_bfloat16;

static inline __device__ float bf16_to_float(BF16 value) {
  return __bfloat162float(value);
}

static inline __device__ BF16 bf16_from_float(float value) {
  return __float2bfloat16_rn(value);
}

static inline BF16 bf16_host_from_float(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16) & 1U);
  uint16_t packed = (uint16_t)(bits >> 16);
  BF16 result;
  std::memcpy(&result, &packed, sizeof(result));
  return result;
}

static inline float bf16_host_to_float(BF16 value) {
  uint16_t packed;
  std::memcpy(&packed, &value, sizeof(packed));
  uint32_t bits = ((uint32_t)packed) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

__device__ void bf16_decode_coords(int index, const int* shape, int rank, int* coords) {
  for (int dim = rank - 1; dim >= 0; dim--) {
    coords[dim] = shape[dim] == 0 ? 0 : index % shape[dim];
    if (shape[dim] != 0) index /= shape[dim];
  }
}

__device__ int bf16_coords_offset(const int* coords, const int* shape, int rank) {
  int offset = 0;
  for (int dim = 0; dim < rank; dim++) offset = offset * shape[dim] + coords[dim];
  return offset;
}

__global__ void bf16_direct_layout_kernel(
  const BF16* input, BF16* output, const int* metadata, int n
) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;
  int rank = metadata[0];
  int operation = metadata[1];
  int dim = metadata[2];
  int argument = metadata[3];
  const int* input_shape = metadata + 4;
  const int* output_shape = metadata + 8;
  const int* axes = metadata + 12;
  int output_coords[4] = {0, 0, 0, 0};
  int input_coords[4] = {0, 0, 0, 0};
  bf16_decode_coords(index, output_shape, rank, output_coords);
  switch (operation) {
    case 0:
      for (int out_dim = 0; out_dim < rank; out_dim++)
        input_coords[axes[out_dim]] = output_coords[out_dim];
      break;
    case 1:
      for (int i = 0; i < rank; i++) input_coords[i] = output_coords[i];
      input_coords[dim] += argument;
      break;
    case 2:
      for (int i = 0; i < rank; i++)
        input_coords[i] = input_shape[i] == 1 ? 0 : output_coords[i];
      break;
    case 3:
      for (int i = 0; i < rank; i++) input_coords[i] = output_coords[i];
      input_coords[dim] /= argument;
      break;
    case 4:
      for (int i = 0; i < rank; i++) input_coords[i] = output_coords[i];
      break;
  }
  output[index] = input[bf16_coords_offset(input_coords, input_shape, rank)];
}

extern "C" int cuda_bf16_direct_layout(
  const void* input, void* output, const int* metadata, int n
) {
  if (n == 0) return 0;
  bf16_direct_layout_kernel<<<(n + 255) / 256, 256>>>(
    (const BF16*)input, (BF16*)output, metadata, n
  );
  return (int)cudaGetLastError();
}

__global__ void bf16_gather_kernel(
  const BF16* input, const int* indices, BF16* output, int n
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = input[indices[i]];
}

extern "C" int cuda_bf16_gather(
  const void* input, const int* indices, void* output, int n
) {
  if (n == 0) return 0;
  bf16_gather_kernel<<<(n + 255) / 256, 256>>>(
    (const BF16*)input, indices, (BF16*)output, n
  );
  return (int)cudaGetLastError();
}

__global__ void bf16_concat2_kernel(
  const BF16* left, const BF16* right, BF16* output,
  int outer, int left_width, int right_width, int inner
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int width = left_width + right_width;
  int total = outer * width * inner;
  if (i >= total) return;
  int inner_index = i % inner;
  int outer_index = i / (width * inner);
  int column = (i / inner) % width;
  if (column < left_width) {
    output[i] = left[(outer_index * left_width + column) * inner + inner_index];
  } else {
    int right_column = column - left_width;
    output[i] = right[(outer_index * right_width + right_column) * inner + inner_index];
  }
}

extern "C" int cuda_bf16_concat2(
  const void* left, const void* right, void* output,
  int outer, int left_width, int right_width, int inner
) {
  int total = outer * (left_width + right_width) * inner;
  if (total == 0) return 0;
  bf16_concat2_kernel<<<(total + 255) / 256, 256>>>(
    (const BF16*)left, (const BF16*)right, (BF16*)output,
    outer, left_width, right_width, inner
  );
  return (int)cudaGetLastError();
}

extern "C" int cuda_bf16_batched_matmul(
  void* handle, void* left, void* right, void* output,
  int batch, int m, int k, int p, int transpose_right
) {
  if (batch == 0) return 0;
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasStatus_t status = cublasGemmStridedBatchedEx(
    (cublasHandle_t)handle,
    transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N,
    p, m, k, &alpha,
    right, CUDA_R_16BF, transpose_right ? k : p, (long long)k * p,
    left, CUDA_R_16BF, k, (long long)m * k,
    &beta,
    output, CUDA_R_16BF, p, (long long)m * p,
    batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
  );
  return (status == CUBLAS_STATUS_SUCCESS) ? 0 : (int)status;
}

extern "C" int cuda_bf16_linear(
  void* handle, void* left, void* right, void* output,
  int rows, int input_size, int output_size
) {
  return cuda_bf16_batched_matmul(
    handle, left, right, output, 1, rows, input_size, output_size, 1
  );
}

__global__ void bf16_rms_norm_kernel(
  const BF16* input, const BF16* weight, BF16* output, int width, int rows, float eps
) {
  int row = blockIdx.x;
  if (row >= rows) return;
  int base = row * width;
  int tid = threadIdx.x;
  __shared__ float partial[256];
  __shared__ float scale;
  float sum = 0.0f;
  for (int i = tid; i < width; i += blockDim.x) {
    float value = bf16_to_float(input[base + i]);
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) partial[tid] += partial[tid + stride];
    __syncthreads();
  }
  if (tid == 0) scale = sqrtf(partial[0] / width + eps);
  __syncthreads();
  for (int i = tid; i < width; i += blockDim.x)
    output[base + i] = bf16_from_float(bf16_to_float(input[base + i]) / scale * bf16_to_float(weight[i]));
}

extern "C" int cuda_bf16_rms_norm(
  const void* input, const void* weight, void* output, int width, int rows, float eps
) {
  if (rows == 0) return 0;
  bf16_rms_norm_kernel<<<rows, 256>>>(
    (const BF16*)input, (const BF16*)weight, (BF16*)output, width, rows, eps
  );
  return (int)cudaGetLastError();
}

__global__ void bf16_softmax_kernel(const BF16* input, BF16* output, int width, int rows) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  int base = row * width;
  float maximum = bf16_to_float(input[base]);
  for (int i = 1; i < width; i++)
    maximum = fmaxf(maximum, bf16_to_float(input[base + i]));
  float total = 0.0f;
  for (int i = 0; i < width; i++) {
    float value = expf(bf16_to_float(input[base + i]) - maximum);
    output[base + i] = bf16_from_float(value);
    total += value;
  }
  for (int i = 0; i < width; i++)
    output[base + i] = bf16_from_float(bf16_to_float(output[base + i]) / total);
}

extern "C" int cuda_bf16_softmax(const void* input, void* output, int width, int rows) {
  if (rows == 0) return 0;
  bf16_softmax_kernel<<<(rows + 31) / 32, 32>>>(
    (const BF16*)input, (BF16*)output, width, rows
  );
  return (int)cudaGetLastError();
}

__global__ void bf16_silu_kernel(const BF16* input, BF16* output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float value = bf16_to_float(input[i]);
    output[i] = bf16_from_float(value / (1.0f + expf(-value)));
  }
}

extern "C" int cuda_bf16_silu(const void* input, void* output, int n) {
  if (n == 0) return 0;
  bf16_silu_kernel<<<(n + 255) / 256, 256>>>(
    (const BF16*)input, (BF16*)output, n
  );
  return (int)cudaGetLastError();
}

__global__ void bf16_mul_kernel(const BF16* left, const BF16* right, BF16* output, int n, int right_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    output[i] = bf16_from_float(bf16_to_float(left[i]) * bf16_to_float(right[i % right_size]));
}

extern "C" int cuda_bf16_mul(const void* left, const void* right, void* output, int n, int right_size) {
  if (n == 0) return 0;
  bf16_mul_kernel<<<(n + 255) / 256, 256>>>(
    (const BF16*)left, (const BF16*)right, (BF16*)output, n, right_size
  );
  return (int)cudaGetLastError();
}

__global__ void bf16_add_kernel(const BF16* left, const BF16* right, BF16* output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = bf16_from_float(bf16_to_float(left[i]) + bf16_to_float(right[i]));
}

extern "C" int cuda_bf16_add(const void* left, const void* right, void* output, int n) {
  if (n == 0) return 0;
  bf16_add_kernel<<<(n + 255) / 256, 256>>>(
    (const BF16*)left, (const BF16*)right, (BF16*)output, n
  );
  return (int)cudaGetLastError();
}

__global__ void bf16_scale_kernel(const BF16* input, BF16* output, int n, float scalar) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = bf16_from_float(bf16_to_float(input[i]) * scalar);
}

extern "C" int cuda_bf16_scale(const void* input, void* output, int n, float scalar) {
  if (n == 0) return 0;
  bf16_scale_kernel<<<(n + 255) / 256, 256>>>(
    (const BF16*)input, (BF16*)output, n, scalar
  );
  return (int)cudaGetLastError();
}

__global__ void bf16_masked_fill_kernel(
  const BF16* input, const BF16* mask, BF16* output, int n, int mask_size, BF16 value
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = bf16_to_float(mask[i % mask_size]) != 0.0f ? value : input[i];
}

extern "C" int cuda_bf16_masked_fill(
  const void* input, const void* mask, void* output, int n, int mask_size, float value
) {
  if (n == 0) return 0;
  bf16_masked_fill_kernel<<<(n + 255) / 256, 256>>>(
    (const BF16*)input, (const BF16*)mask, (BF16*)output, n, mask_size,
    bf16_host_from_float(value)
  );
  return (int)cudaGetLastError();
}

__global__ void bf16_argmax_kernel(const BF16* input, BF16* output, int rows, int width) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  int base = row * width;
  int selected = 0;
  float best = bf16_to_float(input[base]);
  for (int i = 1; i < width; i++) {
    float value = bf16_to_float(input[base + i]);
    if (value > best) {
      best = value;
      selected = i;
    }
  }
  output[row] = bf16_from_float((float)selected);
}

extern "C" int cuda_bf16_argmax(const void* input, void* output, int rows, int width) {
  if (rows == 0) return 0;
  bf16_argmax_kernel<<<(rows + 31) / 32, 32>>>(
    (const BF16*)input, (BF16*)output, rows, width
  );
  return (int)cudaGetLastError();
}

extern "C" int cuda_bf16_to_host(const void* input, float* output, int n) {
  BF16* host = (BF16*)malloc(n * sizeof(BF16));
  if (!host) return 1;
  cudaError_t err = cudaMemcpy(host, input, n * sizeof(BF16), cudaMemcpyDeviceToHost);
  if (err == cudaSuccess) {
    for (int i = 0; i < n; i++) output[i] = bf16_host_to_float(host[i]);
  }
  free(host);
  return (int)err;
}
