/*
This file is part of the Indigo2 benchmark suite version 1.0.

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

typedef unsigned int data_type;
#include "mis_edge_cpp.h"
typedef std::atomic<unsigned char> shared_t;
typedef std::atomic<int> idx_t;

static void init(const ECLgraph& g, const int* const sp, data_type* const priority, shared_t* const status, shared_t* const status_n, bool* const lost, int* const wl1, idx_t& wlsize)
{
  // initialize arrays
  for (int v = 0; v < g.nodes; v++)
  {
    priority[v] = hash(v + 712313887);
    status[v] = undecided;
    status_n[v] = undecided;
    lost[v] = false;
  }
  wlsize = 0;
  for (int e = 0; e < g.edges; e++)
  {
    // initialize worklist
    if (sp[e] < g.nlist[e]) {
      wl1[wlsize++] = e;
    }
  }
}

static void mis(const ECLgraph& g, const int* const sp, const data_type* const priority, shared_t* const status, shared_t* const status_n, bool* const lost, const int* const wl1, const int wl1size, int* const wl2, idx_t& wl2size, const int threadID, const int threadCount)
{
  const int N = wl1size;
  const int begEdge = threadID * (long)N / threadCount;
  const int endEdge = (threadID + 1) * (long)N / threadCount;
  for (int w = begEdge; w < endEdge; w++) {
    int e = wl1[w];
    const int src = sp[e];
    const int dst = g.nlist[e];
    // if one is included, exclude the other
    if (status[src] == included) {
      status_n[dst] = excluded;
    }
    else if (status[dst] == included) {
      status_n[src] = excluded;
    }
    // if neither included nor excluded -> mark lower priority node as lost
    else if (status[src] != excluded && status[dst] != excluded) {
      if (priority[src] < priority[dst]) {
        lost[src] = true;
      } else {
        lost[dst] = true;
      }
    }
  }
}

static void mis_vertex_pass(const ECLgraph& g, const int* const sp, data_type* const priority, shared_t* const status, shared_t* const status_n, bool* const lost, const int* const wl1, const int wl1size, int* const wl2, idx_t& wl2size, const int threadID, const int threadCount)
{
  const int begEdge = threadID * (long)wl1size / threadCount;
  const int endEdge = (threadID + 1) * (long)wl1size / threadCount;

  // go over all vertexes
  for (int w = begEdge; w < endEdge; w++) {
    const int e = wl1[w];
    const int src = sp[e];
    const int dst = g.nlist[e];
    // if src node won
    if (lost[src] == false) {
      if (status[src] == undecided) {
        // and is undecided -> include
        status_n[src] = included;
      }
    }
    // if dst node won
    if (lost[dst] == false) {
      if (status[dst] == undecided) {
        // and is undecided -> include
        status_n[dst] = included;
      }
    }
    if (status[src] == undecided || status[dst] == undecided) {
      // if either node is still undecided, keep edge in WL
      wl2[wl2size++] = e;
    }
  }
}

static void mis_last_pass(shared_t* const status, const int size, const int threadID, const int threadCount)
{
  const int begNode = threadID * (long)size / threadCount;
  const int endNode = (threadID + 1) * (long)size / threadCount;

  for (int v = begNode; v < endNode; v++) {
    if (status[v] == undecided) {
      status[v] = included;
    }
  }
}

static double CPPmis_edge(const ECLgraph& g, const int* const sp, data_type* const priority, unsigned char* const status_orig, const int threadCount)
{
  shared_t* status = (shared_t*)status_orig;
  shared_t* status_new = new shared_t [g.nodes];
  bool* lost = new bool [g.nodes];
  const int size = std::max(g.edges, g.nodes);
  int* wl1 = new int [size];
  int* wl2 = new int [size];
  idx_t wl1size;
  idx_t wl2size;
  std::thread threadHandles[threadCount];

  init(g, sp, priority, status, status_new, lost, wl1, wl1size);

  timeval beg, end;
  gettimeofday(&beg, NULL);

  int iter = 0;
  do {
    iter++;
    wl2size = 0;

    // edge pass
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(mis, g, sp, priority, status, status_new, lost, wl1, wl1size.load(), wl2, std::ref(wl2size), i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }

    // merge pass
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(updateFromWorklist, g, sp, (unsigned char*)status, (unsigned char*)status_new, wl1, wl1size.load(), i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }

    // vertex pass
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(mis_vertex_pass, g, sp, priority, status, status_new, lost, wl1, wl1size.load(), wl2, std::ref(wl2size), i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }

    // merge pass
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(updateFromWorklist, g, sp, (unsigned char*)status, (unsigned char*)status_new, wl1, wl1size.load(), i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }
    std::fill(lost, lost + g.nodes, false);
    std::swap(wl1, wl2);
    wl1size = wl2size.load();
  } while (wl1size > 0);

  for (int i = 0; i < threadCount; ++i) {
    threadHandles[i] = std::thread(mis_last_pass, status, g.nodes, i, threadCount);
  }
  for (int i = 0; i < threadCount; ++i) {
    threadHandles[i].join();
  }

  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;

  // determine and print set size
  int cnt = 0;
  for (int v = 0; v < g.nodes; v++) {
    if (status[v] == included) cnt++;
  }
  printf("iterations: %d,  elements in set: %d (%.1f%%)\n", iter, cnt, 100.0 * cnt / g.nodes);

  delete [] status_new;
  delete [] lost;
  delete [] wl1;
  delete [] wl2;
  return runtime;
}
