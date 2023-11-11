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

#include <cuda/atomic>
typedef cuda::atomic<int> flag_t;
typedef int data_type;
static const int ThreadsPerBlock = 512;

#include "mis_edge_cuda.h"

static __global__ void init(const ECLgraph g, const int* const sp, data_type* const priority, flag_t* const status, flag_t* const lost, int* const wl1, int* const wlsize)
{
  // initialize arrays
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < g.nodes)
  {
    priority[v] = hash(v + 712313887);
    status[v] = undecided;
    lost[v] = 0;
  }
  if (v < g.edges)
  {
    // initialize worklist
    if (sp[v] < g.nlist[v]) {
      wl1[atomicAdd(wlsize, 1)] = v;
    }
  }
}

static __global__ void mis(const ECLgraph g, const int* const sp, const data_type* const priority, flag_t* const status, flag_t* const lost, const int* const wl1, const int wl1size)
{
  // go over all edges in wl1
  int w = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (w < wl1size) {

    int e = wl1[w];
    const int src = sp[e];
    const int dst = g.nlist[e];
    const int srcStatus = status[src].load();
    const int dstStatus = status[dst].load();

    // if one is included, exclude the other
    if (srcStatus == included) {
      status[dst].store(excluded);
    }
    else if (dstStatus == included) {
      status[src].store(excluded);
    } else if (srcStatus == undecided && dstStatus == undecided) {
      // if both undecided -> mark lower as lost
      if (priority[src] < priority[dst]) {
        lost[src].store(1);
      } else {
        lost[dst].store(1);
      }
    }
  }
}

static __global__ void mis_vertex_pass(const ECLgraph g, const int* const sp, data_type* const priority, flag_t* const status, flag_t* const lost, const int* const wl1, const int wl1size, int* const wl2, int* const wl2size, const int iter, int* const time)
{
  // go over all edges in wl1 and check if lost
  int w = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (w < wl1size) {

    const int e = wl1[w];
    const int src = sp[e];
    const int dst = g.nlist[e];
    const int srcStatus = status[src].load();
    const int dstStatus = status[dst].load();

    // if src won
    if (lost[src] == 0) {
      if (srcStatus == undecided) {
        // and is undecided -> include
        status[src].store(included);
      }
    }
    // if dst won
    if (lost[dst] == 0) {
      if (dstStatus == undecided) {
        // and is undecided -> include
        status[dst].store(included);
      }
    }
    // if either is still undecided, keep it in WL
    if (srcStatus == undecided || dstStatus == undecided) {
      if (atomicMax(&time[e], iter) < iter) {
        wl2[atomicAdd(wl2size, 1)] = e;
      }
    }
  }
}

static __global__ void mis_last_pass(flag_t* const status, const int size)
{
  int w = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (w < size) {
    if (status[w] == undecided)
    {
      status[w] = included;
    }
  }
}

static double GPUmis_edge(const ECLgraph& g, const int* const sp, data_type* const priority, int* const status)
{
  data_type* d_priority;
  if (cudaSuccess != cudaMalloc((void **)&d_priority, g.nodes * sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_priority\n");
  flag_t* d_status;
  if (cudaSuccess != cudaMalloc((void **)&d_status, g.nodes * sizeof(flag_t))) fprintf(stderr, "ERROR: could not allocate d_status\n");
  flag_t* d_lost;
  if (cudaSuccess != cudaMalloc((void **)&d_lost, g.nodes * sizeof(flag_t))) fprintf(stderr, "ERROR: could not allocate d_lost\n");

  int* d_wl1;
  if (cudaSuccess != cudaMalloc((void **)&d_wl1, std::max(g.edges, g.nodes) * sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl1\n");
  int* d_wl1size;
  if (cudaSuccess != cudaMalloc((void **)&d_wl1size, sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl1size\n");
  int* d_wl2;
  if (cudaSuccess != cudaMalloc((void **)&d_wl2, std::max(g.edges, g.nodes) * sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl2\n");
  int* d_wl2size;
  if (cudaSuccess != cudaMalloc((void **)&d_wl2size, sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl2size\n");
  int* d_time;
  if (cudaSuccess != cudaMalloc((void **)&d_time, sizeof(int) * g.edges)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  cudaMemset(d_time, 0, sizeof(int) * g.edges);
  int wlsize;
  cudaMemset(d_wl1size, 0, sizeof(int));

  init<<<(g.edges + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(g, sp, d_priority, d_status, d_lost, d_wl1, d_wl1size);

  if (cudaSuccess != cudaMemcpy(&wlsize, d_wl1size, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of wlsize from device d_wl1size failed\n");

  timeval beg, end;
  gettimeofday(&beg, NULL);

  int iter = 0;
  do {
    iter++;
    cudaMemset(d_wl2size, 0, sizeof(int));
    const int blocks = (wlsize + ThreadsPerBlock - 1) / ThreadsPerBlock;

    // edge pass
    mis<<<blocks, ThreadsPerBlock>>>(g, sp, d_priority, d_status, d_lost, d_wl1, wlsize);

    // vertex pass
    mis_vertex_pass<<<blocks, ThreadsPerBlock>>>(g, sp, d_priority, d_status, d_lost, d_wl1, wlsize, d_wl2, d_wl2size, iter, d_time);

    if (cudaSuccess != cudaMemcpy(&wlsize, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost)) { fprintf(stderr, "ERROR: copying of wlsize from device failed\n"); break; }
    cudaMemset(d_lost, 0, g.nodes * sizeof(flag_t));
    std::swap(d_wl1, d_wl2);
    std::swap(d_wl1size, d_wl2size);
  } while (wlsize > 0);

  const int blocks = (g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock;
  // include all remaining nodes that have no edges
  mis_last_pass<<<blocks, ThreadsPerBlock>>>(d_status, g.nodes);

  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;

  CheckCuda();
  if (cudaSuccess != cudaMemcpy(status, d_status, g.nodes * sizeof(flag_t), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of status from device failed\n");

  // determine and print set size
  int cnt = 0;
  for (int v = 0; v < g.nodes; v++) {
    if (status[v] == included) cnt++;
  }
  printf("iterations: %d,  elements in set: %d (%.1f%%)\n", iter, cnt, 100.0 * cnt / g.nodes);

  cudaFree(d_status);
  cudaFree(d_priority);
  cudaFree(d_lost);
  cudaFree(d_wl1);
  cudaFree(d_wl1size);
  cudaFree(d_wl2);
  cudaFree(d_wl2size);
  return runtime;
}
