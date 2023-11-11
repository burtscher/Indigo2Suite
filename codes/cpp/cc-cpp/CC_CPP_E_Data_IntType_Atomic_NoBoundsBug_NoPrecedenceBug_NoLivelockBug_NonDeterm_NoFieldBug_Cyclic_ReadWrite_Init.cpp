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
#include "cc_edge_cpp.h"
typedef std::atomic<data_type> shared_t;
typedef std::atomic<int> idx_t;

static void init(shared_t* const label, const int size, const ECLgraph g, int* const wl1, idx_t& wlsize, idx_t* const time)
{
  int idx = 0;
  // initialize label array
  for (int v = 0; v < size; v++) {
    label[v] = v;

    for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
      wl1[idx] = i;
      idx++;
    }
  }
  wlsize = idx;
}

static void cc_edge_data(const ECLgraph g, const int* const sp, shared_t* const label, const int* const wl1, const int wl1size, int* const wl2, idx_t& wl2size, const int iter, idx_t* const time, const int threadID, const int threadCount)
{
  const int N = wl1size;
  const int top = N;
  for (int idx = threadID; idx < top; idx += threadCount) {
    const int e = wl1[idx];
    const int src = sp[e];
    const int dst = g.nlist[e];
    const data_type new_label = label[src];
    data_type d = label[dst];
    if (d > new_label) {
      label[dst] = new_label;
      if (atomicMax(&time[e], iter) != iter) {
        wl2[wl2size++] = e;
      }
      for (int j = g.nindex[dst]; j < g.nindex[dst + 1]; j++) {
        if (atomicMax(&time[j], iter) != iter) {
          wl2[wl2size++] = j;
        }
      }
    }
  }
}

static double CPUcc_edge(const ECLgraph g, data_type* const label_orig, const int* const sp, const int threadCount)
{
  shared_t* label = (shared_t*)label_orig;
  const int size = std::max(g.edges, g.nodes);
  int* wl1 = new int [size];
  int* wl2 = new int [size];
  idx_t* time = new idx_t [g.edges];
  idx_t wl1size;
  idx_t wl2size;
  std::thread threadHandles[threadCount];

  init(label, g.nodes, g, wl1, wl1size, time);
  std::fill((int*)time, (int*)time + g.edges, 0);

  timeval start, end;
  gettimeofday(&start, NULL);

  // iterate until no more changes
  int iter = 0;
  do {
    iter++;
    wl2size = 0;

    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(cc_edge_data, g, sp, label, wl1, wl1size.load(), wl2, std::ref(wl2size), iter, time, i, threadCount);
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
  delete [] time;
  return runtime;
}
