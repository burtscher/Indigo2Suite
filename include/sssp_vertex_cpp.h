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
#include <queue>
#include <sys/time.h>
#include "ECLgraph.h"
#include <climits>
#include <numeric>
#include <thread>
#include <atomic>

const data_type maxval = std::numeric_limits<data_type>::max();
using pair = std::pair<int, int>;

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

static double CPUsssp_vertex(const int src, const ECLgraph& g, data_type* const dist, const int threadCount);

static void CPUserialDijkstra(const int src, const ECLgraph& g, data_type* const dist)
{
  // initialize dist array
  for (int i = 0; i < g.nodes; i++) dist[i] = maxval;
  dist[src] = 0;

  // set up priority queue with just source node in it
  std::priority_queue< std::pair<int, int> > pq;
  pq.push(std::make_pair(0, src));
  while (pq.size() > 0) {
    // process closest vertex
    const int v = pq.top().second;
    pq.pop();
    const data_type dv = dist[v];
    // visit outgoing neighbors
    for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
      const int n = g.nlist[i];
      const data_type d = dv + g.eweight[i];
      // check if new lower distance found
      if (d < dist[n]) {
        dist[n] = d;
        pq.push(std::make_pair(-d, n));
      }
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
  printf("sssp topology-driven CPP (%s)\n", __FILE__);
  if (argc != 4 && argc != 5) {fprintf(stderr, "USAGE: %s input_file_name source_node_number verify thread_count(optional)\n", argv[0]); exit(-1);}

  // process command line
  ECLgraph g = readECLgraph(argv[1]);
  if (g.eweight == NULL) {
    printf("Generating weights.\n");
    g.eweight = new int [g.edges];
    for (int i = 0; i < g.nodes; i++) {
      for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
        const int nei = g.nlist[j];
        g.eweight[j] = 1 + ((i * nei) % g.nodes);
          if (g.eweight[j] < 0) g.eweight[j] = -g.eweight[j];
      }
    }
  }
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  const int source = atoi(argv[2]);
  if ((source < 0) || (source >= g.nodes)) {fprintf(stderr, "ERROR: source_node_number must be between 0 and %d\n", g.nodes); exit(-1);}
  printf("source: %d\n", source);
  const int runveri = atoi(argv[3]);
  if ((runveri != 0) && (runveri != 1)) {
    printf("has to be 0 (turn off) or 1 (turn on) verification");
  }
  
  int threadCount = std::thread::hardware_concurrency(); //defaults to max threads
  if(argc == 5)
    if(const int countInt = atoi(argv[4])) //checks for valid int
      threadCount = countInt;             //takes optional argument for thread count
  printf("Threads: %d\n", threadCount);

  // allocate memory
  data_type* const distance = new data_type [g.nodes];

  // launch kernel
  const int runs = 3;
  double runtimes [runs];
  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPUsssp_vertex(source, g, distance, threadCount);
  }
  const double med = median(runtimes, runs);
  printf("runtime: %.6fs\n", med);
  printf("Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);

  // print result
  int maxnode = 0;
  for (int v = 1; v < g.nodes; v++) {
    if (distance[maxnode] < distance[v]) maxnode = v;
  }
  printf("vertex %d has maximum distance %d\n", maxnode, distance[maxnode]);

  // compare solutions
  if (runveri) {
    data_type* const verify = new data_type [g.nodes];
    CPUserialDijkstra(source, g, verify);
    for (int v = 0; v < g.nodes; v++) {
      if (distance[v] != verify[v]) {fprintf(stderr, "ERROR: verification failed for node %d: %d   instead of %d\n", v, distance[v], verify[v]); exit(-1);}
    }
    printf("verification passed\n\n");
    delete [] verify;
  } else {
    printf("turn off verification\n\n");
  }
  // free memory
  delete [] distance;
  freeECLgraph(g);
  return 0;
}
