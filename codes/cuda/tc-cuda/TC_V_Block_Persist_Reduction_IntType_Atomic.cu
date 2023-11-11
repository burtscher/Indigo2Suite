/*
This file is part of the Indigo2 benchmark suite version 0.9.

Copyright (c) 2023, Yiqian Liu, Noushin Azami, Avery Vanausdal, and Martin Burtscher

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

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

URL: The latest version of this code is available at https://cs.txstate.edu/~burtscher/research/Indigo2Suite/.

Publication: This work is described in detail in the following paper.

Yiqian Liu, Noushin Azami, Avery Vanausdal, and Martin Burtscher. "Choosing the Best Parallelization and Implementation Styles for Graph Analytics Codes: Lessons Learned from 1106 Programs." Proceedings of the 2023 ACM/IEEE International Conference for High Performance Computing, Networking, Storage, and Analysis. November 2023.
*/

typedef int data_type;
typedef int basic_t;
static const int WS = 32;
static const int ThreadsPerBlock = 512;
static const int Device = 0;
#include "tc_vertex_cuda.h"
static __global__ void d_triCounting(data_type* g_count, const int nodes, const int* const __restrict__ nindex, const int* const __restrict__ nlist)
{
  __shared__ int s_buffer[WS];
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;
  basic_t count = 0;
  const int idx = blockIdx.x;
  for (int v = idx; v < nodes; v += gridDim.x) {
    const int beg1 = nindex[v];
    const int end1 = nindex[v + 1];
    int start1 = end1;
    while ((beg1 < start1) && (v < nlist[start1 - 1])) start1--;
    for (int j = start1 + threadIdx.x; j < end1; j += ThreadsPerBlock){
      const int u = nlist[j];
      const int beg2 = nindex[u];
      const int end2 = nindex[u + 1];
      int start2 = end2;
      while ((beg2 < start2) && (u < nlist[start2 - 1])) start2--;
      count += (basic_t)d_common(j + 1, end1, start2, end2, nlist);
    }
  }
  // warp reduction
  count += __shfl_down_sync(~0, count, 16);
  count += __shfl_down_sync(~0, count, 8);
  count += __shfl_down_sync(~0, count, 4);
  count += __shfl_down_sync(~0, count, 2);
  count += __shfl_down_sync(~0, count, 1);
  if (lane == 0) s_buffer[warp] = count;
  __syncthreads();
  // block reduction
  if (warp == 0) {
    int val = s_buffer[lane];
    val += __shfl_down_sync(~0, val, 16);
    val += __shfl_down_sync(~0, val, 8);
    val += __shfl_down_sync(~0, val, 4);
    val += __shfl_down_sync(~0, val, 2);
    val += __shfl_down_sync(~0, val, 1);
    if (lane == 0) atomicAdd(g_count, val);
  }
}
static double GPUtc_vertex(basic_t &count, const int nodes, const int* const nindex, const int* const nlist)
{
  data_type* d_count;
  if (cudaSuccess != cudaMalloc((void **)&d_count, sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_goagain\n");
  timeval start, end;
  const int ThreadsBound = GPUinfo(Device);
  const int blocks = ThreadsBound / ThreadsPerBlock;
  count = 0;
  gettimeofday(&start, NULL);
  if (cudaSuccess != cudaMemcpy(d_count, &count, sizeof(data_type), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of go_again to device failed\n");
  d_triCounting<<<blocks, ThreadsPerBlock>>>(d_count, nodes, nindex, nlist);
  if (cudaSuccess != cudaMemcpy(&count, d_count, sizeof(data_type), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of go_again from device failed\n");
  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);
  cudaFree(d_count);
  return (end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0);
}
