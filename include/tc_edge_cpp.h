#include <algorithm>
#include <sys/time.h>
#include <thread>
#include <atomic>
#include "ECLgraph.h"

static double CPUtc_edge(basic_t &count, const int edges, const int* const nindex, const int* const nlist, const int* const sp, const int threadCount);

static inline int h_common(const int beg1, const int end1, const int beg2, const int end2, const int* const __restrict__ nlist)
{
  int common = 0;
  int pos1 = beg1;
  int pos2 = beg2;
  while ((pos1 < end1) && (pos2 < end2)) {
    while ((pos1 < end1) && (nlist[pos1] < nlist[pos2])) pos1++;
    if (pos1 < end1) {
      while ((pos2 < end2) && (nlist[pos2] < nlist[pos1])) pos2++;
      if ((pos2 < end2) && (nlist[pos1] == nlist[pos2])) {
        pos1++;
        pos2++;
        common++;
      } else {
        pos1++;
      }
    }
  }
  return common;
}

static basic_t h_triCounting(const int nodes, const int* const __restrict__ nindex, const int* const __restrict__ nlist)
{
  basic_t count = 0;

  for (int v = 0; v < nodes; v++) {
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
      count += h_common(j + 1, end1, start2, end2, nlist);
    }
  }
  return count;
}

struct CPUTimer
{
  timeval beg, end;
  CPUTimer() {}
  ~CPUTimer() {}
  void start() {gettimeofday(&beg, NULL);}
  double elapsed() {gettimeofday(&end, NULL); return end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;}
};

static double median(double array[], const int n)
{
  double median = 0;
  std::sort(array, array + n);
  if (n % 2 == 0) median = (array[(n - 1) / 2] + array[n / 2]) / 2.0;
  else median = array[n / 2];
  return median;
}

static inline bool find(const int target, const int beg, const int end, const int* const __restrict__ nlist)
{
  int left = beg;
  int right = end;
  while (left <= right) {
    int middle = (left + right) / 2;
    if (nlist[middle] == target) return true;
    if (nlist[middle] < target) left = middle + 1;
    else right = middle - 1;
  }
  return false;
}

int main(int argc, char* argv [])
{
  printf("Triangle counting edge-centric CPP (%s)\n", __FILE__);

  if (argc != 3 && argc != 4) {printf("USAGE: %s input_graph verify thread_count(optional)\n\n", argv[0]);  exit(-1);}

  // read input
  ECLgraph g = readECLgraph(argv[1]);
  printf("graph: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  printf("avg degree: %.2f\n", 1.0 * g.edges / g.nodes);

  // info only
  int mdeg = 0;
  for (int v = 0; v < g.nodes; v++) {
    mdeg = std::max(mdeg, g.nindex[v + 1] - g.nindex[v]);
  }
  printf("max degree: %d\n", mdeg);

  // check if sorted
  for (int v = 0; v < g.nodes; v++) {
    for (int i = g.nindex[v] + 1; i < g.nindex[v + 1]; i++) {
      if (g.nlist[i - 1] >= g.nlist[i]) {
        printf("ERROR: adjacency list not sorted or contains self edge\n");
        exit(-1);
      }
    }
  }

  // create starting point array
  int* const sp = new int [g.edges];
  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      sp[j] = i;
    }
  }
  
  int threadCount = std::thread::hardware_concurrency(); //defaults to max threads
  if(argc == 4)
    if(const int countInt = atoi(argv[3])) //checks for valid int
      threadCount = countInt;             //takes optional argument for thread count
  printf("Threads: %d\n\n", threadCount);

  // allocate memory
  basic_t count = 0;

  // launch kernel
  const int runs = 3;
  double runtimes [runs];

  for (int i = 0; i < runs; i++) {
  	runtimes[i] = CPUtc_edge(count, g.edges, g.nindex, g.nlist, sp, threadCount);
  }

  const double med = median(runtimes, runs);
  printf("CPP runtime: %.6f s\n", med);
  printf("CPP Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);

  // verify
  const int verify = atoi(argv[2]);
  if ((verify != 0) && (verify != 1)) {
    printf("has to be 0 (turn off) or 1 (turn on) verification");
  }
  if (verify) {
  	timeval start, end;
    gettimeofday(&start, NULL);
    basic_t h_count = h_triCounting(g.nodes, g.nindex, g.nlist);
    gettimeofday(&end, NULL);
    // printf("CPU runtime: %.6fs\n", end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0);
    if (h_count != count) printf("ERROR: host %ld device %ld", h_count, count);
    else printf("the pattern occurs %ld times\n\n", count);
  }

  // clean up
  delete [] sp;
  freeECLgraph(g);
  return 0;
}

template <typename T>
static inline T atomicAdd(T* addr, T val)
{
  T old = ((std::atomic<T>*)addr)->load();
  while (!(((std::atomic<T>*)addr)->compare_exchange_weak(old, old + val))) {}
  return old;
}

template <typename T>
static inline T atomicAdd(std::atomic<T>* addr, T val)
{
  T old = addr->load();
  while (!(addr->compare_exchange_weak(old, old + val))) {}
  return old;
}

