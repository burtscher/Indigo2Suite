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
#include "bfs_vertex_cpp.h"
typedef std::atomic<data_type> shared_t;
typedef std::atomic<int> idx_t;

static void init(const int src, shared_t* const dist, const int size, int* const wl1, idx_t* const wlsize)
{
  // initialize dist array
  for (int v = 0; v < size; v++) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist[v] = temp;
  }
  // initialize worklist
  wl1[0] = src;
  *wlsize = 1;
}

static void bfs_vertex_data(const ECLgraph g, shared_t* const dist, const int* const wl1, const int wl1size, int* const wl2, idx_t& wl2size, const int iter, const int threadID, const int threadCount)
{
  const int N = wl1size;
  const int top = N;
  for (int idx = threadID; idx < top; idx += threadCount) {
    const int src = wl1[idx];
    const data_type s = dist[src];
    const int beg = g.nindex[src];
    const int end = g.nindex[src + 1];

    if (s != maxval) {
      for (int i = beg; i < end; i++) {
        const int dst = g.nlist[i];
        const data_type new_dist = s + 1;
        if (atomicMin(&dist[dst], new_dist) > new_dist) {
          wl2[wl2size++] = dst;
        }
      }
    }
  }
}
static double CPPbfs_vertex(const int src, const ECLgraph& g, data_type* const distance, const int threadCount)
{
  shared_t* dist = (shared_t*)distance;
  const int size = std::max(g.edges, g.nodes);
  int* wl1 = new int [size];
  int* wl2 = new int [size];
  idx_t wl1size = 0;
  idx_t wl2size;
  std::thread threadHandles[threadCount];

  init(src, dist, g.nodes, wl1, &wl1size);

  timeval start, end;
  gettimeofday(&start, NULL);

  // iterate until no more changes
  int iter = 0;
  do {
    iter++;
    wl2size = 0;

    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(bfs_vertex_data, g, dist, wl1, wl1size.load(), wl2, std::ref(wl2size), iter, i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }

    std::swap(wl1, wl2);
    wl1size = wl2size.load();
  } while (wl1size > 0);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  delete [] wl1;
  delete [] wl2;
  return runtime;
}
