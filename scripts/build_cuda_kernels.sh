#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="$PROJECT/moonxi-net-gpu/cuda/cuda_kernels.cu"
OUTPUT_DIR="${CUDA_BUILD_DIR:-$PROJECT/_build/cuda}"
OBJECT="$OUTPUT_DIR/cuda_kernels.o"
LIBRARY="$OUTPUT_DIR/libmoonxi_cuda_kernels.a"
STAMP="$OUTPUT_DIR/build.sha256"

if [[ -n "${NVCC:-}" ]]; then
  NVCC_BIN="$NVCC"
elif [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
  NVCC_BIN="$CUDA_HOME/bin/nvcc"
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  NVCC_BIN=/usr/local/cuda/bin/nvcc
else
  NVCC_BIN="$(command -v nvcc || true)"
fi

if [[ -z "$NVCC_BIN" || ! -x "$NVCC_BIN" ]]; then
  echo "nvcc not found; set NVCC or CUDA_HOME" >&2
  exit 1
fi

DEFAULT_FLAGS="--generate-code arch=compute_75,code=[sm_75,compute_75] --generate-code arch=compute_80,code=[sm_80,compute_80] --generate-code arch=compute_86,code=[sm_86,compute_86] --generate-code arch=compute_89,code=[sm_89,compute_89] --generate-code arch=compute_120,code=[sm_120,compute_120] -Wno-deprecated-gpu-targets"
FLAGS="${NVCC_FLAGS:-$DEFAULT_FLAGS}"
read -r -a FLAG_ARGS <<< "$FLAGS"

INCLUDE_ARGS=()
if [[ -n "${CUDNN_INCLUDE:-}" ]]; then
  INCLUDE_ARGS+=("-I$CUDNN_INCLUDE")
fi

mkdir -p "$OUTPUT_DIR"
FINGERPRINT="$(
  {
    sha256sum "$SOURCE"
    printf '%s\n' "$NVCC_BIN" "$FLAGS" "${CUDNN_INCLUDE:-}"
  } | sha256sum | cut -d' ' -f1
)"

if [[ -f "$LIBRARY" && -f "$STAMP" ]] && [[ "$(<"$STAMP")" == "$FINGERPRINT" ]]; then
  echo "CUDA kernels unchanged: $LIBRARY"
  exit 0
fi

"$NVCC_BIN" -std=c++17 -Xcompiler -fPIC "${INCLUDE_ARGS[@]}" \
  "${FLAG_ARGS[@]}" -c "$SOURCE" -o "$OBJECT"
"${AR:-ar}" rcs "$LIBRARY" "$OBJECT"
printf '%s\n' "$FINGERPRINT" > "$STAMP"
echo "Built CUDA kernels: $LIBRARY"
