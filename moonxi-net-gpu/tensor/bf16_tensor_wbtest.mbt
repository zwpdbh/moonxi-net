// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
fn assert_bf16_values(
  value : Bf16Tensor,
  expected : Array[Float],
) -> Unit raise {
  let actual = value.to_host()
  assert_eq(actual.length(), expected.length())
  for i in 0..<expected.length() {
    if (actual[i] - expected[i]).abs() >= 0.02F {
      println("index \{i}: expected \{expected[i]}, got \{actual[i]}")
      assert_true(false)
    }
  }
}

///|
test "BF16 tensors execute Qwen3 operations on CUDA" {
  @cuda.gpu_init(0)
  let input = Bf16Tensor::from_host(
    FixedArray::from_array([1.0F, 2.0F, 3.0F, 4.0F]),
    [2, 2],
  )
  assert_bf16_values(input, [1.0F, 2.0F, 3.0F, 4.0F])
  assert_bf16_values(input.permute([1, 0]), [1.0F, 3.0F, 2.0F, 4.0F])
  assert_bf16_values(input.narrow(dim=0, start=1, length=1), [3.0F, 4.0F])
  assert_bf16_values(Bf16Tensor::concat([input, input], dim=0), [
    1.0F, 2.0F, 3.0F, 4.0F, 1.0F, 2.0F, 3.0F, 4.0F,
  ])
  assert_bf16_values(input.repeat_interleave(repeats=2, dim=1), [
    1.0F, 1.0F, 2.0F, 2.0F, 3.0F, 3.0F, 4.0F, 4.0F,
  ])

  let weight = Bf16Tensor::from_host(
    FixedArray::from_array([1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F]),
    [3, 2],
  )
  assert_bf16_values(input.linear(weight), [1.0F, 2.0F, 3.0F, 3.0F, 4.0F, 7.0F])
  assert_bf16_values(input.batched_matmul(input), [7.0F, 10.0F, 15.0F, 22.0F])
  assert_bf16_values(input.softmax_last(), [0.2689F, 0.7311F, 0.2689F, 0.7311F])
  assert_bf16_values(input.silu(), [0.7311F, 1.7616F, 2.8577F, 3.9281F])
  assert_bf16_values(
    input.rms_norm(
      Bf16Tensor::from_host(FixedArray::from_array([1.0F, 1.0F]), [2]),
      eps=0.0F,
    ),
    [0.6325F, 1.2649F, 0.8485F, 1.1314F],
  )
  assert_bf16_values(
    input.masked_fill(
      Bf16Tensor::from_host(FixedArray::from_array([0.0F, 1.0F]), [1, 2]),
      value=-1.0F,
    ),
    [1.0F, -1.0F, 3.0F, -1.0F],
  )
  assert_bf16_values(input.argmax_last(), [1.0F, 1.0F])
  assert_bf16_values(input + input, [2.0F, 4.0F, 6.0F, 8.0F])
  assert_bf16_values(input.scale(2.0F), [2.0F, 4.0F, 6.0F, 8.0F])
  assert_bf16_values(input.mul_elem(input), [1.0F, 4.0F, 9.0F, 16.0F])
  assert_bf16_values(input.persistent(), [1.0F, 2.0F, 3.0F, 4.0F])
  let embeddings = Bf16Tensor::from_host(
    FixedArray::from_array([1.0F, 2.0F, 3.0F, 4.0F]),
    [2, 2],
  )
  assert_bf16_values(embeddings.embedding([1, 0]), [3.0F, 4.0F, 1.0F, 2.0F])
}

///|
test "BF16 operations match Qwen3 ranks" {
  @cuda.gpu_init(0)
  let input = Bf16Tensor::from_host(
    FixedArray::from_array([1.0F, 2.0F, 3.0F, 4.0F]),
    [1, 2, 2],
  )
  let weight = Bf16Tensor::from_host(
    FixedArray::from_array([1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F]),
    [3, 2],
  )
  assert_bf16_values(input.linear(weight), [1.0F, 2.0F, 3.0F, 3.0F, 4.0F, 7.0F])

  let left = Bf16Tensor::from_host(
    FixedArray::from_array([1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F]),
    [2, 2, 2],
  )
  let right = Bf16Tensor::from_host(
    FixedArray::from_array([1.0F, 0.0F, 0.0F, 1.0F, 2.0F, 0.0F, 0.0F, 2.0F]),
    [2, 2, 2],
  )
  assert_bf16_values(left.batched_matmul(right), [
    1.0F, 2.0F, 3.0F, 4.0F, 10.0F, 12.0F, 14.0F, 16.0F,
  ])

  let rank4 = Bf16Tensor::from_host(
    FixedArray::makei(16, i => Float::from_int(i)),
    [1, 2, 2, 4],
  )
  assert_bf16_values(
    rank4.permute([0, 2, 1, 3]).narrow(dim=1, start=1, length=1),
    [4.0F, 5.0F, 6.0F, 7.0F, 12.0F, 13.0F, 14.0F, 15.0F],
  )
}
