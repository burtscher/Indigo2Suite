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

typedef float score_type;
#include "pr_cpp.h"
typedef std::atomic<double> error_type;
typedef std::atomic<score_type> shared_t;

static void incomingCalc(const ECLgraph g, score_type* outgoing_contrib, shared_t* incoming_total, const int threadID, const int threadCount)
{
  const int N = g.nodes;
  const int top = N;
  for (int i = threadID; i < top; i += threadCount) {
    const score_type outgoing = outgoing_contrib[i];
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      const int nei = g.nlist[j];
      atomicAdd(&incoming_total[nei], outgoing);
    }
  }
}

static void errorCalc(const int nodes, error_type& error, score_type* outgoing_contrib, score_type* incoming_total, score_type* scores, const score_type base_score, const int threadID, const int threadCount)
{
  double local_error = 0;
  const int top = nodes;
  for (int i = threadID; i < top; i += threadCount) {
    score_type incoming = incoming_total[i];
    score_type old_score = scores[i];
    const score_type value = base_score + kDamp * incoming;
    scores[i] = value;
    local_error += fabs(value - old_score);
  }
  atomicAdd(&error, local_error);
}

static double PR_CPU(const ECLgraph g, score_type *scores, int* degree, const int threadCount)
{
  const score_type base_score = (1.0f - kDamp) / (score_type)g.nodes;
  score_type* outgoing_contrib = (score_type*)malloc(g.nodes * sizeof(score_type));
  score_type* incoming_total = (score_type*)malloc(g.nodes * sizeof(score_type));
  std::thread threadHandles[threadCount];

  int iter;
  timeval start, end;
  gettimeofday(&start, NULL);

  for (iter = 0; iter < MAX_ITER; iter++) {
    error_type error = 0;
    for (int i = 0; i < g.nodes; i++) {
      outgoing_contrib[i] = scores[i] / degree[i];
      incoming_total[i] = 0;
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(incomingCalc, g, outgoing_contrib, (shared_t*)incoming_total, i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(errorCalc, g.nodes, std::ref(error), outgoing_contrib, incoming_total, scores, base_score, i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }
    if (error < EPSILON) break;
  }

  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  if (iter < MAX_ITER) iter++;
  printf("CPU iterations = %d.\n", iter);

  return runtime;
}
