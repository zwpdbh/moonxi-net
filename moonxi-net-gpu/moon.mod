name = "chnlkw/moonxi-net-gpu"

version = "0.1.1"

readme = "README.md"

repository = "https://github.com/moonxi-net/moonxi-net"

license = "MIT"

keywords = [
  "deep-learning",
  "cuda",
  "cudnn",
  "resnet",
  "neural-network",
  "autograd",
  "gpu",
]

preferred_target = "native"

description = "GPU backend: CUDA/cuDNN tensor operations and training for moonxi-net"

options(
  "--moonbit-unstable-prebuild": "build_config.js",
)

import {
  "chnlkw/moonxi-net@0.1.1",
  "brickfrog/moontrace@0.14.0",
  "moonbitlang/x@0.5.5",
}
