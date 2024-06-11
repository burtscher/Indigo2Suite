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

#include <algorithm>
#include <set>
#include <sys/time.h>
#include "ECLgraph.h"
#include <limits>
#include <thread>
#include <atomic>

static double CPUcc_vertex(const ECLgraph g, data_type* label, const int threadCount);

static void verify(const int v, const int id, const int* const __restrict__ nidx, const int* const __restrict__ nlist, data_type* const __restrict__ nstat, const int nodes)
{
  if (nstat[v] < nodes) {
    if (nstat[v] != id) {fprintf(stderr, "ERROR: found incorrect ID value\n\n");  exit(-1);}
    nstat[v] = nodes;
    for (int i = nidx[v]; i < nidx[v + 1]; i++) {
      verify(nlist[i], id, nidx, nlist, nstat, nodes);
    }
  }
}

static double median(double array[], const int n)
{
  double median = 0;
  std::sort(array, array + n);
  if (n % 2 == 0) median = (array[(n - 1) / 2] + array[n / 2]) / 2.0;
  else median = array[n / 2];
  return median;
}

int main(int argc, char* argv[])
{
  printf("cc vertex-based CPP (%s)\n", __FILE__);
  if (argc != 3 && argc != 4) {fprintf(stderr, "USAGE: %s input_file_name verify thread_count(optional)\n", argv[0]); exit(-1);}

  // process command line
  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  const int runveri = atoi(argv[2]);
  if ((runveri != 0) && (runveri != 1)) {
    printf("has to be 0 (turn off) or 1 (turn on) verification");
  }
  
  int threadCount = std::thread::hardware_concurrency(); //defaults to max threads
  if(argc == 4)
    if(const int countInt = atoi(argv[3])) //checks for valid int
      threadCount = countInt;             //takes optional argument for thread count
  printf("Threads: %d\n", threadCount);

  // allocate memory
  data_type* const label = new data_type [g.nodes];

  // cc
  const int runs = 3;
  double runtimes [runs];
  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPUcc_vertex(g, label, threadCount);
  }
  const double med = median(runtimes, runs);
  printf("CPP runtime: %.6fs\n", med);
  printf("CPP Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);

  // print result
  std::set<int> s1;
  for (int v = 0; v < g.nodes; v++) {
    s1.insert(label[v]);
  }
  printf("number of connected components: %d\n", s1.size());

  // compare solutions
  if (runveri) {
    /* verification code (may need extra runtime stack space due to deep recursion) */

    for (int v = 0; v < g.nodes; v++) {
      for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
        if (label[g.nlist[i]] != label[v]) {fprintf(stderr, "ERROR: found adjacent nodes in different components\n\n");  exit(-1);}
      }
    }

    for (int v = 0; v < g.nodes; v++) {
      if (label[v] >= g.nodes) {fprintf(stderr, "ERROR: found sentinel number\n\n");  exit(-1);}
    }

    std::set<int> s2;
    int count = 0;
    for (int v = 0; v < g.nodes; v++) {
      if (label[v] < g.nodes) {
        count++;
        s2.insert(label[v]);
        verify(v, label[v], g.nindex, g.nlist, label, g.nodes);
      }
    }
    if (s1.size() != s2.size()) {fprintf(stderr, "ERROR: number of components do not match\n\n");  exit(-1);}
    if (s1.size() != count) {fprintf(stderr, "ERROR: component IDs are not unique\n\n");  exit(-1);}

    printf("verification passed\n\n");
  } else {
    printf("turn off verification\n\n");
  }

  // free memory
  delete [] label;
  freeECLgraph(g);
  return 0;
}

template <typename T>
static inline T atomicMin(T* addr, T val)
{
  T oldv = ((std::atomic<T>*)addr)->load();
  while (oldv > val && !(((std::atomic<T>*)addr)->compare_exchange_weak(oldv, val))) {}
  return oldv;
}

template <typename T>
static inline T atomicMax(T* addr, T val)
{
  T oldv = ((std::atomic<T>*)addr)->load();
  while (oldv < val && !(((std::atomic<T>*)addr)->compare_exchange_weak(oldv, val))) {}
  return oldv;
}

template <typename T>
static inline T atomicMin(std::atomic<T>* addr, T val)
{
  T oldv = addr->load();
  while (oldv > val && !(addr->compare_exchange_weak(oldv, val))) {}
  return oldv;
}

template <typename T>
static inline T atomicMax(std::atomic<T>* addr, T val)
{
  T oldv = addr->load();
  while (oldv < val && !(addr->compare_exchange_weak(oldv, val))) {}
  return oldv;
}
