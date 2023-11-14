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
typedef std::atomic<bool> flag_t;

static void init(data_type* const priority, shared_t* const status, shared_t* const status_n, bool* const lost, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++)
  {
    priority[v] = hash(v + 712313887);
    status[v] = undecided;
    status_n[v] = undecided;
    lost[v] = false;
  }
}

static void mis(const ECLgraph& g, const int* const sp, data_type* const priority, shared_t* const status, shared_t* const status_n, bool* const lost, const int threadID, const int threadCount)
{
  const int N = g.edges;
  const int top = N;
  for (int e = threadID; e < top; e += threadCount) {
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
      // mark lower priority node as lost
      if (priority[src] < priority[dst]) {
        lost[src] = true;
      } else {
        lost[dst] = true;
      }
    }
  }
}

static void mis_vertex_pass(shared_t* const status, shared_t* const status_n, bool* const lost, const int size, bool& goagain, const int threadID, const int threadCount)
{
  const int top = size;
  // go over all vertexes
  for (int v = threadID; v < top; v += threadCount) {
    if (lost[v] == false) {
      if(status[v] == undecided) {
        status_n[v] = included;
      }
    }
    else {
      goagain = true;
      lost[v] = false; //reset lost flag
    }
  }
}

static void mis_last_pass(shared_t* const status, const int size, const int threadID, const int threadCount)
{
  const int top = size;

  for (int v = threadID; v < top; v += threadCount) {
    if (status[v] == undecided)
    {
      status[v] = included;
    }
  }
}

static double CPPmis_edge(const ECLgraph& g, const int* const sp, data_type* const priority, unsigned char* const status_orig, const int threadCount)
{
  shared_t* status = (shared_t*)status_orig;
  shared_t* status_new = new shared_t [g.nodes];
  bool* lost = new bool [g.nodes];
  std::thread threadHandles[threadCount];

  init(priority, status, status_new, lost, g.nodes);

  timeval beg, end;
  gettimeofday(&beg, NULL);

  bool goagain;
  int iter = 0;
  do {
    iter++;
    goagain = false;

    // edge pass
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(mis, g, sp, priority, status, status_new, lost, i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }

    // merge pass
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(updateUndecided, (unsigned char*)status, (unsigned char*)status_new, g.nodes, i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }

    // vertex pass
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(mis_vertex_pass, status, status_new, lost, g.nodes, std::ref(goagain), i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }

    // merge pass
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(updateUndecided, (unsigned char*)status, (unsigned char*)status_new, g.nodes, i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }

  } while (goagain);

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
  return runtime;
}
