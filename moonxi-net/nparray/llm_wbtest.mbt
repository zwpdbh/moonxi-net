// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
fn assert_llm_shape(value : NpArray, expected : Array[Int32]) -> Unit raise {
  assert_eq(value.shape.shape.length(), expected.length())
  for i in 0..<expected.length() {
    assert_eq(value.shape.shape[i], expected[i])
  }
}

///|
test "permute and narrow reindex arbitrary tensors" {
  let input = NpArray::from2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let transposed = input.permute([1, 0])
  assert_llm_shape(transposed, [3, 2])
  assert_true(np_eq(transposed, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]))
  let selected = transposed.select(dim=0, index=1)
  assert_llm_shape(selected, [2])
  assert_true(np_eq(selected, [2.0, 5.0]))
  let expanded = selected.unsqueeze(dim=0).expand([2, 2])
  assert_true(np_eq(expanded, [2.0, 5.0, 2.0, 5.0]))
}

///|
test "concat and repeat_interleave support cache-style dimensions" {
  let left = NpArray::from2d([[1.0, 2.0]])
  let right = NpArray::from2d([[3.0, 4.0]])
  let joined = NpArray::concat([left, right], dim=0)
  assert_llm_shape(joined, [2, 2])
  assert_true(np_eq(joined, [1.0, 2.0, 3.0, 4.0]))
  let repeated = joined.repeat_interleave(repeats=2, dim=1)
  assert_llm_shape(repeated, [2, 4])
  assert_true(np_eq(repeated, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]))
}

///|
test "batched matmul keeps independent batch values" {
  let left = NpArray::from_host(
    FixedArray::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    [2, 2, 2],
  )
  let right = NpArray::from_host(
    FixedArray::from_array([1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]),
    [2, 2, 2],
  )
  let result = left.batched_matmul(right)
  assert_llm_shape(result, [2, 2, 2])
  assert_true(np_eq(result, [1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0]))
}

///|
test "linear preserves leading dimensions" {
  let input = NpArray::from_host(FixedArray::from_array([1.0, 2.0, 3.0, 4.0]), [
    1, 2, 2,
  ])
  let weight = NpArray::from2d([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
  let result = input.linear(weight)
  assert_llm_shape(result, [1, 2, 3])
  assert_true(np_eq(result, [1.0, 2.0, 3.0, 3.0, 4.0, 7.0]))
}

///|
test "softmax is stable and normalized by row" {
  let input = NpArray::from2d([[0.0, 0.0], [10000.0, 10001.0]])
  let result = input.softmax_last()
  assert_true((result.data[0] - 0.5F).abs() < 0.000001F)
  assert_true((result.data[1] - 0.5F).abs() < 0.000001F)
  assert_true((result.data[2] - 0.26894143F).abs() < 0.000001F)
  assert_true((result.data[3] - 0.7310586F).abs() < 0.000001F)
}

///|
test "embedding uses integer IDs without float labels" {
  let weight = NpArray::from2d([[1.0, 2.0], [3.0, 4.0]])
  let result = weight.embedding([1, 0])
  assert_llm_shape(result, [2, 2])
  assert_true(np_eq(result, [3.0, 4.0, 1.0, 2.0]))
}

///|
test "rms_norm silu masking and argmax match reference values" {
  let input = NpArray::from2d([[1.0, 2.0]])
  let normalized = input.rms_norm(NpArray::from1d([1.0, 1.0]), eps=0.0F)
  assert_true((normalized.data[0] - 0.6324555F).abs() < 0.000001F)
  assert_true((normalized.data[1] - 1.264911F).abs() < 0.000001F)
  assert_true(
    (NpArray::from1d([1.0]).silu().data[0] - 0.7310586F).abs() < 0.000001F,
  )
  let masked = input.masked_fill(NpArray::from2d([[0.0, 1.0]]), value=-1.0F)
  assert_true(np_eq(masked, [1.0, -1.0]))
  let ids = NpArray::from2d([[1.0, 3.0, 2.0], [9.0, 4.0, 5.0]]).argmax_last()
  assert_llm_shape(ids, [2])
  assert_true(np_eq(ids, [1.0, 0.0]))
}
