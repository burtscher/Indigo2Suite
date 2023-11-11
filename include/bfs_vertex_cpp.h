#include <climits>
#include <algorithm>
#include <queue>
#include <sys/time.h>
#include <thread>
#include <atomic>
#include "ECLgraph.h"

const data_type maxval = std::numeric_limits<data_type>::max();

using pair = std::pair<int, int>;

static double CPPbfs_vertex(const int src, const ECLgraph& g, data_type* const dist, const int threadCount);

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
      const int d = dv + 1;
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
  printf("bfs vertex-based CPP (%s)\n", __FILE__);
  if (argc != 5 && argc != 6) {fprintf(stderr, "USAGE: %s input_file_name source_node_number verify runs thread_count(optional)\n", argv[0]); exit(-1);}

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
  const int runs = 3;
  int threadCount = std::thread::hardware_concurrency(); //defaults to max threads
  if(argc == 6)
    if(const int countInt = atoi(argv[5])) //checks for valid int
      threadCount = countInt;             //takes optional argument for thread count
  printf("Threads: %d\n", threadCount);

  // allocate memory
  data_type* const distance = new data_type [g.nodes];

  // launch kernel
  double runtimes [runs];
  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPPbfs_vertex(source, g, distance, threadCount);
  // compare solutions
  if (runveri) {
    data_type* const verify = new data_type [g.nodes];
    CPUserialDijkstra(source, g, verify);
    for (int v = 0; v < g.nodes; v++) {
      if (distance[v] != verify[v]) {fprintf(stderr, "ERROR: verification failed for node %d: %d instead of %d\n", v, distance[v], verify[v]); exit(-1);}
    }
    printf("verification passed\n\n");
    delete [] verify;
  } else {
    printf("turn off verification\n\n");
  }
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

  // free memory
  delete [] distance;
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
