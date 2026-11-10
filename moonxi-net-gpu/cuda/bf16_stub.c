#include <moonbit.h>
#include <stdint.h>

extern int cuda_bf16_direct_layout(const void*, void*, const int*, int);
extern int cuda_bf16_gather(const void*, const int*, void*, int);
extern int cuda_bf16_concat2(const void*, const void*, void*, int, int, int, int);
extern int cuda_bf16_batched_matmul(void*, void*, void*, void*, int, int, int, int, int);
extern int cuda_bf16_linear(void*, void*, void*, void*, int, int, int);
extern int cuda_bf16_rms_norm(const void*, const void*, void*, int, int, float);
extern int cuda_bf16_softmax(const void*, void*, int, int);
extern int cuda_bf16_silu(const void*, void*, int);
extern int cuda_bf16_mul(const void*, const void*, void*, int, int);
extern int cuda_bf16_add(const void*, const void*, void*, int);
extern int cuda_bf16_scale(const void*, void*, int, float);
extern int cuda_bf16_masked_fill(const void*, const void*, void*, int, int, float);
extern int cuda_bf16_argmax(const void*, void*, int, int);
extern int cuda_bf16_to_host(const void*, float*, int);

#define BF16_WRITER(name, params, args) \
  MOONBIT_FFI_EXPORT int32_t name params { return args; }

BF16_WRITER(moonbit_bf16_direct_layout,
  (void* input, void* output, void* metadata, int32_t n),
  cuda_bf16_direct_layout(input, output, (const int*)metadata, n))
BF16_WRITER(moonbit_bf16_gather,
  (void* input, void* indices, void* output, int32_t n),
  cuda_bf16_gather(input, (const int*)indices, output, n))
BF16_WRITER(moonbit_bf16_concat2,
  (void* left, void* right, void* output, int32_t outer, int32_t left_width, int32_t right_width, int32_t inner),
  cuda_bf16_concat2(left, right, output, outer, left_width, right_width, inner))
BF16_WRITER(moonbit_bf16_batched_matmul,
  (void* handle, void* left, void* right, void* output, int32_t batch, int32_t m, int32_t k, int32_t p, int32_t transpose_right),
  cuda_bf16_batched_matmul(handle, left, right, output, batch, m, k, p, transpose_right))
BF16_WRITER(moonbit_bf16_linear,
  (void* handle, void* left, void* right, void* output, int32_t rows, int32_t input_size, int32_t output_size),
  cuda_bf16_linear(handle, left, right, output, rows, input_size, output_size))
BF16_WRITER(moonbit_bf16_rms_norm,
  (void* input, void* weight, void* output, int32_t width, int32_t rows, float eps),
  cuda_bf16_rms_norm(input, weight, output, width, rows, eps))
BF16_WRITER(moonbit_bf16_softmax,
  (void* input, void* output, int32_t width, int32_t rows),
  cuda_bf16_softmax(input, output, width, rows))
BF16_WRITER(moonbit_bf16_silu,
  (void* input, void* output, int32_t n),
  cuda_bf16_silu(input, output, n))
BF16_WRITER(moonbit_bf16_mul,
  (void* left, void* right, void* output, int32_t n, int32_t right_size),
  cuda_bf16_mul(left, right, output, n, right_size))
BF16_WRITER(moonbit_bf16_add,
  (void* left, void* right, void* output, int32_t n),
  cuda_bf16_add(left, right, output, n))
BF16_WRITER(moonbit_bf16_scale,
  (void* input, void* output, int32_t n, float scalar),
  cuda_bf16_scale(input, output, n, scalar))
BF16_WRITER(moonbit_bf16_masked_fill,
  (void* input, void* mask, void* output, int32_t n, int32_t mask_size, float value),
  cuda_bf16_masked_fill(input, mask, output, n, mask_size, value))
BF16_WRITER(moonbit_bf16_argmax,
  (void* input, void* output, int32_t rows, int32_t width),
  cuda_bf16_argmax(input, output, rows, width))
BF16_WRITER(moonbit_bf16_to_host,
  (void* input, float* output, int32_t n),
  cuda_bf16_to_host(input, output, n))
