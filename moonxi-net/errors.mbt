// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
#derive(Debug)
pub(all) suberror ShapeError {
  ShapeError(String)
}

///|
#derive(Debug)
pub(all) suberror IoError {
  IoError(String)
}

///|
#derive(Debug)
pub(all) suberror CudaError {
  CudaError(String)
}

///|
#derive(Debug)
pub(all) suberror UnsupportedError {
  UnsupportedError(String)
}
