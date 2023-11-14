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
#include <limits>

const data_type maxval = std::numeric_limits<data_type>::max();
using pair = std::pair<int, int>;

template <typename T>
static inline T atomicRead(T* const addr)
{
  T ret;
  #pragma omp atomic read
  ret = *addr;
  return ret;
}

template <typename T>
static inline void atomicWrite(T* const addr, const T val)
{
  #pragma omp atomic write
  *addr = val;
}

template <typename T>
static inline T critical_min(T* addr, T val)
{
  T oldv;
  #pragma omp critical
  {
    oldv = *addr;
    if (oldv > val) {
      *addr = val;
    }
  }
  return oldv;
}

template <typename T>
static inline T critical_max(T* addr, T val)
{
  T oldv;
  #pragma omp critical
  {
    oldv = *addr;
    if (oldv < val) {
      *addr = val;
    }
  }
  return oldv;
}

template <typename T>
static inline T fetch_and_add(T* addr)
{
  T old;
  #pragma omp atomic capture
  {
    old = *addr;
    (*addr)++;
  }
  return old;
}

static double CPUbfs_vertex(const int src, const ECLgraph& g, data_type* dist);

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
      const data_type d = dv + 1;
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
  printf("bfs topology-driven OMP (%s)\n", __FILE__);
  if (argc != 4) {fprintf(stderr, "USAGE: %s input_file_name source_node_number verify\n", argv[0]); exit(-1);}

  // process command line
  ECLgraph g = readECLgraph(argv[1]);
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

  // allocate memory
  data_type* const distance = new data_type [g.nodes];

  // bfs
  const int runs = 3;
  double runtimes [runs];
  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPUbfs_vertex(source, g, distance);
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
