const path = require("node:path");

const cudaDirectory = path.resolve(__dirname, "../_build/cuda");
const output = {
  link_configs: [
    {
      package: "chnlkw/moonxi-net-gpu/cuda",
      link_search_paths: [
        cudaDirectory,
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
      ],
      link_libs: [
        "moonxi_cuda_kernels",
        "cuda",
        "cudart",
        "cublas",
        "cudnn",
        "stdc++",
      ],
    },
  ],
};

process.stdout.write(JSON.stringify(output));
