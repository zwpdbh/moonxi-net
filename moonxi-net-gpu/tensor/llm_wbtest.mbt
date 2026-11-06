// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
fn assert_gpu_values(value : GpuTensor, expected : Array[Float]) -> Unit raise {
  let actual = value.to_host()
  assert_eq(actual.length(), expected.length())
  for i in 0..<expected.length() {
    if (actual[i] - expected[i]).abs() >= 0.0001F {
      println("index \{i}: expected \{expected[i]}, got \{actual[i]}")
      assert_true(false)
    }
  }
}

///|
test "LLM operations execute on CUDA" {
  @cuda.gpu_init(0)
  let input = GpuTensor::from_host(
    FixedArray::from_array([1.0F, 2.0F, 3.0F, 4.0F]),
    [2, 2],
  )
  let transposed = input.permute([1, 0])
  assert_gpu_values(transposed, [1.0F, 3.0F, 2.0F, 4.0F])
  assert_gpu_values(transposed.narrow(dim=0, start=1, length=1), [2.0F, 4.0F])
  assert_gpu_values(GpuTensor::concat([input, input], dim=0), [
    1.0F, 2.0F, 3.0F, 4.0F, 1.0F, 2.0F, 3.0F, 4.0F,
  ])
  assert_gpu_values(input.repeat_interleave(repeats=2, dim=1), [
    1.0F, 1.0F, 2.0F, 2.0F, 3.0F, 3.0F, 4.0F, 4.0F,
  ])

  let weight = GpuTensor::from_host(
    FixedArray::from_array([1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F]),
    [3, 2],
  )
  assert_gpu_values(input.linear(weight), [1.0F, 2.0F, 3.0F, 3.0F, 4.0F, 7.0F])
  assert_gpu_values(input.batched_matmul(input), [7.0F, 10.0F, 15.0F, 22.0F])
  assert_gpu_values(input.softmax_last(), [
    0.26894143F, 0.7310586F, 0.26894143F, 0.7310586F,
  ])
  assert_gpu_values(input.silu(), [
    0.7310586F, 1.7615942F, 2.8577225F, 3.9280553F,
  ])
  assert_gpu_values(
    input.rms_norm(
      GpuTensor::from_host(FixedArray::from_array([1.0F, 1.0F]), [2]),
      eps=0.0F,
    ),
    [0.6324555F, 1.264911F, 0.84852815F, 1.131371F],
  )
  assert_gpu_values(
    input.masked_fill(
      GpuTensor::from_host(FixedArray::from_array([0.0F, 1.0F]), [1, 2]),
      value=-1.0F,
    ),
    [1.0F, -1.0F, 3.0F, -1.0F],
  )
  assert_gpu_values(input.argmax_last(), [1.0F, 1.0F])
  let embeddings = GpuTensor::from_host(
    FixedArray::from_array([1.0F, 2.0F, 3.0F, 4.0F]),
    [2, 2],
  )
  assert_gpu_values(embeddings.embedding([1, 0]), [3.0F, 4.0F, 1.0F, 2.0F])
}
