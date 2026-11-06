#include <moonbit.h>
#include <stdint.h>

extern int cuda_llm_gather(const float*, const int*, float*, int);
extern int cuda_llm_batched_matmul(void*, float*, float*, float*, int, int, int, int, int);
extern int cuda_llm_softmax(const float*, float*, int, int);
extern int cuda_llm_silu(const float*, float*, int);
extern int cuda_llm_rms_norm(const float*, const float*, float*, int, int, float);
extern int cuda_llm_masked_fill(const float*, const float*, float*, int, int, float);
extern int cuda_llm_mul_broadcast(const float*, const float*, float*, int, int);
extern int cuda_free_buffer(void*);

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_llm_gather(void* input, void* indices, void* output, int32_t n) {
    return cuda_llm_gather((const float*)input, (const int*)indices, (float*)output, n);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_llm_batched_matmul(
    void* handle, void* left, void* right, void* output,
    int32_t batch, int32_t m, int32_t k, int32_t p, int32_t transpose_right
) {
    return cuda_llm_batched_matmul(handle, (float*)left, (float*)right, (float*)output,
                                   batch, m, k, p, transpose_right);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_llm_softmax(void* input, void* output, int32_t width, int32_t rows) {
    return cuda_llm_softmax((const float*)input, (float*)output, width, rows);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_llm_silu(void* input, void* output, int32_t n) {
    return cuda_llm_silu((const float*)input, (float*)output, n);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_llm_rms_norm(
    void* input, void* weight, void* output, int32_t width, int32_t rows, float eps
) {
    return cuda_llm_rms_norm((const float*)input, (const float*)weight, (float*)output,
                             width, rows, eps);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_llm_masked_fill(
    void* input, void* mask, void* output, int32_t n, int32_t mask_size, float value
) {
    return cuda_llm_masked_fill((const float*)input, (const float*)mask, (float*)output,
                                n, mask_size, value);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_llm_mul_broadcast(
    void* left, void* right, void* output, int32_t n, int32_t right_size
) {
    return cuda_llm_mul_broadcast((const float*)left, (const float*)right,
                                  (float*)output, n, right_size);
}

extern int cuda_llm_concat2(const float*, const float*, float*, int, int, int, int);

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_llm_concat2(
    void* left, void* right, void* output,
    int32_t outer, int32_t left_width, int32_t right_width, int32_t inner
) {
    return cuda_llm_concat2((const float*)left, (const float*)right, (float*)output,
                            outer, left_width, right_width, inner);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_free(void* ptr) {
    return cuda_free_buffer(ptr);
}
