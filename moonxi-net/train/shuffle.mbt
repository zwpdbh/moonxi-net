// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
pub fn fisher_yates_shuffle(arr : Array[Int32]) -> Array[Int32] {
  let result = arr.copy()
  let rng = @random.Rand()
  let mut i = result.length() - 1
  while i > 0 {
    let j = rng.int(limit=i + 1)
    let tmp = result[i]
    result[i] = result[j]
    result[j] = tmp
    i -= 1
  }
  result
}
