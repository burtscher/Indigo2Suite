#include <climits>
#include <algorithm>
#include <queue>
#include <sys/time.h>
#include <thread>
#include <atomic>
#include "ECLgraph.h"

const data_type maxval = std::numeric_limits<data_type>::max();

using pair = std::pair<int, int>;

const unsigned char undecided = 0;
const unsigned char included = 1;
const unsigned char excluded = 2;

static double CPPmis_vertex(const ECLgraph& g, data_type* const priority, unsigned char* const status, const int threadCount);

static void verify(ECLgraph& g, unsigned char* const status)
{
  for (int v = 0; v < g.nodes; v++) {
    if ((status[v] != included) && (status[v] != excluded)) {fprintf(stderr, "ERROR: found undecided node\n"); exit(-1);}
    if (status[v] == included) {
      for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
        if (status[g.nlist[i]] == included) {fprintf(stderr, "ERROR: found adjacent nodes in MIS\n"); exit(-1);}
      }
    } else {
      bool flag = true;
      for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
        if (status[g.nlist[i]] == included) {
          flag = false;
          break;
        }
      }
      if (flag) {fprintf(stderr, "ERROR: set is not maximal\n"); exit(-1);}
    }
  }
  printf("verification passed\n");
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
  printf("mis vertex-based CPP (%s)\n", __FILE__);
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
  data_type* const priority = new data_type [g.nodes];
  unsigned char* const status = new unsigned char [g.nodes];

  // launch kernel
  const int runs = 3;
  double runtimes [runs];
  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPPmis_vertex(g, priority, status, threadCount);
    // verify result
    if (runveri) {
      verify(g, status);
    } else {
      printf("turn off verification\n\n");
    }
  }
  const double med = median(runtimes, runs);
  printf("runtime: %.6fs\n", med);
  printf("Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);

  // free memory
  delete [] priority;
  delete [] status;
  freeECLgraph(g);
  return 0;
}

// https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
static unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

static void updateUndecided(unsigned char* const status, unsigned char* const status_n, const int size, const int threadID, const int threadCount)
{
  const int begNode = threadID * (long)size / threadCount;
  const int endNode = (threadID + 1) * (long)size / threadCount;
  
  for (int i = begNode; i < endNode; ++i)
  {
    if (status[i] == undecided)
      status[i] = status_n[i];
  }
}

static void updateFromWorklist(unsigned char* const status, unsigned char* const status_n, const int* const worklist, const int wlsize, const int threadID, const int threadCount)
{
  const int begNode = threadID * (long)wlsize / threadCount;
  const int endNode = (threadID + 1) * (long)wlsize / threadCount;
  
  for (int i = begNode; i < endNode; ++i)
  {
    int v = worklist[i];
    status[v] = status_n[v];
  }
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
