# Odd/Even Loop PyTorch Extension

This directory contains a loadable PyTorch extension for the gfx950 hipBLASLt
custom BF16 GEMM assembly kernel:

```text
Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1_UserArgs_shortname1_gfx950
```

Files:

```text
odd_even_loop.cpp              C++/pybind extension source
odd_even_loop.py               Python loader/wrapper
test_odd_even_loop.py          Correctness and triton.testing.do_bench benchmark
original.s                     Original odd/even split-loop assembly kernel
same_loop_even.s               Modified kernel with all SIMDs forced through WVLoop0
```

HSACO files are generated on demand from the `.s` files into
`torch_ext_build/hsaco/` by `odd_even_loop.py`.

Usage:

```python
import sys
sys.path.insert(0, "/root/development/odd_even_loop")

import torch
import odd_even_loop as ext

M, N, K = 4096, 4096, 8192
a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
b_storage = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
b = b_storage.T

c_original = ext.matmul_original(a, b)
c_same_loop = ext.matmul_same_loop(a, b)

out_original = ext.empty_output(a, b)
out_same_loop = ext.empty_output(a, b)
ext.matmul_original_out(a, b, out_original)
ext.matmul_same_loop_out(a, b, out_same_loop)
```

Benchmark:

```bash
python3 /root/development/odd_even_loop/test_odd_even_loop.py --k 8192
python3 /root/development/odd_even_loop/test_odd_even_loop.py --k 16384
```
