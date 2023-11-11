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

typedef float score_type;
#include "pr_cuda.h"

__global__ void contrib(int nodes, score_type* scores, int* degree, score_type* outgoing_contrib, score_type* incoming_total)
{
  int tid = blockIdx.x;
  for (int src = tid; src < nodes; src += gridDim.x) {
    outgoing_contrib[src] = scores[src] / degree[src];
    incoming_total[src] = 0;
  }
}

__global__ void push(int nodes, const int* const __restrict__ nindex, const int* const __restrict__ nlist, score_type* outgoing_contrib, score_type* incoming_total)
{
  int tid = blockIdx.x;
  for (int src = tid; src < nodes; src += gridDim.x) {
    const int beg = nindex[src];
    const int end = nindex[src + 1];
    const score_type outgoing = outgoing_contrib[src];
    // iterate neighbor list
    for (int i = beg + threadIdx.x; i < end; i += ThreadsPerBlock) {
      int dst = nlist[i];
      atomicAdd(&incoming_total[dst], outgoing);
    }
  }
}

__global__ void compute(int nodes, score_type* scores, score_type* diff, score_type base_score, score_type* incoming_total)
{
  score_type error = 0;
  int tid = blockIdx.x;
  for (int src = tid; src < nodes; src += gridDim.x) {
    score_type old_score = scores[src];
    score_type incoming = incoming_total[src];
    const score_type value = base_score + kDamp * incoming;
    scores[src] = value;
    error = fabs(value - old_score);
    atomicAdd(diff, error);
  }
}

void PR_GPU(const ECLgraph g, score_type *scores, int* degree)
{
  ECLgraph d_g = g;
  int *d_degree;
  score_type *d_scores, *d_sums, *d_contrib, *d_incoming;
  score_type *d_diff, h_diff;
  const int ThreadsBound = GPUinfo(0);
  const int blocks = ThreadsBound / ThreadsPerBlock;
  // allocate device memory
  cudaMalloc((void **)&d_degree, g.nodes * sizeof(int));
  cudaMalloc((void **)&d_scores, g.nodes * sizeof(score_type));
  cudaMalloc((void **)&d_sums, g.nodes * sizeof(score_type));
  cudaMalloc((void **)&d_contrib, g.nodes * sizeof(score_type));
  cudaMalloc((void **)&d_incoming, g.nodes * sizeof(score_type));
  cudaMalloc((void **)&d_diff, sizeof(score_type));
  if (cudaSuccess != cudaMalloc((void **)&d_g.nindex, (g.nodes + 1) * sizeof(int))) fprintf(stderr, "ERROR: could not allocate nindex\n");
  if (cudaSuccess != cudaMalloc((void **)&d_g.nlist, g.edges * sizeof(int))) fprintf(stderr, "ERROR: could not allocate nlist\n");
  // copy data to device
  cudaMemcpy(d_degree, degree, g.nodes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_scores, scores, g.nodes * sizeof(score_type), cudaMemcpyHostToDevice);
  if (cudaSuccess != cudaMemcpy(d_g.nindex, g.nindex, (g.nodes + 1) * sizeof(int), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of index to device failed\n");
  if (cudaSuccess != cudaMemcpy(d_g.nlist, g.nlist, g.edges * sizeof(int), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of nlist to device failed\n");
  const score_type base_score = (1.0f - kDamp) / (score_type)g.nodes;
  // timer
  const int runs = 1;
  timeval start, end;
  double runtimes[runs];
  for (int i = 0; i < runs; i++) {
    int iter = 0;
    gettimeofday(&start, NULL);
    do {
      iter++;
      h_diff = 0;
      if (cudaSuccess != cudaMemcpy(d_diff, &h_diff, sizeof(score_type), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of h_diff to device failed\n");
      contrib<<<blocks, ThreadsPerBlock>>>(g.nodes, d_scores, d_degree, d_contrib, d_incoming);
      push<<<blocks, ThreadsPerBlock>>>(g.nodes, d_g.nindex, d_g.nlist, d_contrib, d_incoming);
      compute<<<blocks, ThreadsPerBlock>>>(g.nodes, d_scores, d_diff, base_score, d_incoming);
      if (cudaSuccess != cudaMemcpy(&h_diff, d_diff, sizeof(score_type), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of d_diff from device failed\n");
    } while (h_diff > EPSILON && iter < MAX_ITER);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    runtimes[i] = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
    printf("GPU iterations = %d.\n", iter);
  }
  const double med = median(runtimes, runs);
  printf("GPU runtime: %.6fs\n\n", med);
  printf("Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);
  if (cudaSuccess != cudaMemcpy(scores, d_scores, g.nodes * sizeof(score_type), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of d_scores from device failed\n");
  cudaFree(d_degree);
  cudaFree(d_scores);
  cudaFree(d_sums);
  cudaFree(d_contrib);
  cudaFree(d_diff);
  return;
}
