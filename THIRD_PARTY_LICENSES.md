# Third-party licenses

This project's overall license is MIT (see `LICENSE`). Parts of the CUDA
rollout/sampling-distribution framework -- including the `Controller<>`
template, `kernelType`/split-vs-single-kernel machinery, the
`GaussianDistributionImpl`/`ColoredNoiseDistributionImpl` sampling
distributions, the `Managed` base class, and the general structure of
`mppi_common.cu`'s rollout kernels -- are adapted from
[MPPI-Generic](https://github.com/ACDSLab/MPPI-Generic) (Georgia Institute of
Technology), used here under the BSD 2-Clause License below. Some files
(e.g. `include/mppi_controller_cuda/utils/managed.cuh`) carry this notice
directly at the top of the file instead of relying solely on this notice.

## MPPI-Generic (BSD 2-Clause License)

```
BSD 2-Clause License

Copyright (c) 2020, Georgia Institute of Technology

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
