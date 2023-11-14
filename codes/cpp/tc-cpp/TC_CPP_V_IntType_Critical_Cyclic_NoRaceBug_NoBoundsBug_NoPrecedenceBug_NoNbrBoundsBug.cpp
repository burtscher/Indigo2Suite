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

#include <mutex>
typedef int basic_t;
#include "tc_vertex_cpp.h"

typedef basic_t shared_t;
std::mutex mutex;

static void triCounting(shared_t& g_count, const int nodes, const int* const nindex, const int* const nlist, const int threadID, const int threadCount)
{
  basic_t count = 0;
  const int top = nodes;
  for (int v = threadID; v < top; v += threadCount) {
    const int beg1 = nindex[v];
    const int end1 = nindex[v + 1];
    int start1 = end1;

    while ((beg1 < start1) && (v < nlist[start1 - 1])) start1--;
    for (int j = start1; j < end1; j++) {
      const int u = nlist[j];
      const int beg2 = nindex[u];
      const int end2 = nindex[u + 1];
      int start2 = end2;
      while ((beg2 < start2) && (u < nlist[start2 - 1])) start2--;
      count += (basic_t)common(j + 1, end1, start2, end2, nlist);
    }
  }
  mutex.lock();
  g_count += count;
  mutex.unlock();
}

static double CPUtc_vertex(basic_t &count_orig, const int nodes, const int* const nindex, const int* const nlist, const int threadCount)
{
  shared_t count = 0;
  std::thread threadHandles[threadCount];

  timeval start, end;
  count = 0;

  gettimeofday(&start, NULL);

  for (int i = 0; i < threadCount; ++i) {
    threadHandles[i] = std::thread(triCounting, std::ref(count), nodes, nindex, nlist, i, threadCount);
  }
  for (int i = 0; i < threadCount; ++i) {
    threadHandles[i].join();
  }

  gettimeofday(&end, NULL);

  count_orig = count;
  return (end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0);
}
