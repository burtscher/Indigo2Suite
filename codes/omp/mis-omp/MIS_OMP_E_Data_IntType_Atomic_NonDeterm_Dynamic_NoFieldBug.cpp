/*
This file is part of the Indigo2 benchmark suite version 0.9.

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
#include "mis_edge_omp.h"

static void init(const ECLgraph& g, const int* const sp, data_type* const priority, unsigned char* const status, bool* const lost, int* const wl1, int &wlsize)
{
  // initialize arrays
  for (int v = 0; v < g.nodes; v++)
  {
    priority[v] = hash(v + 712313887);
    status[v] = undecided;
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

static void mis(const ECLgraph& g, const int* const sp, const data_type* const priority, unsigned char* const status, bool* const lost, const int* const wl1, const int wl1size, int* const wl2, int& wl2size)
{
  #pragma omp parallel for schedule(dynamic)
  for (int w = 0; w < wl1size; w++) {
    // go over all edges in wl1
    int e = wl1[w];
    const int src = sp[e];
    const int dst = g.nlist[e];
    // if one is included, exclude the other
    if (atomicRead(&status[src]) == included) {
      atomicWrite(&status[dst], excluded);
    }
    else if (atomicRead(&status[dst]) == included) {
      atomicWrite(&status[src], excluded);
    }
    // if neither included nor excluded -> mark lower as lost
    else if (atomicRead(&status[src]) != excluded && atomicRead(&status[dst]) != excluded) {
      if (priority[src] < priority[dst]) { //src is lower -> mark lost
      lost[src] = true;
    } else { //dst is lower  -> mark lost
    lost[dst] = true;
  }
}
}
}

static void mis_vertex_pass(const ECLgraph& g, const int* const sp, data_type* const priority, unsigned char* const status, bool* const lost, const int* const wl1, const int wl1size, int* const wl2, int& wl2size)
{
#pragma omp parallel for schedule(dynamic)
for (int w = 0; w < wl1size; w++) {
const int e = wl1[w];
const int src = sp[e];
const int dst = g.nlist[e];
if (lost[src] == false) { // if src won
if (atomicRead(&status[src]) == undecided) {
  // and is undecided -> include
  atomicWrite(&status[src], included);
}
}
if (lost[dst] == false) { // if dst won
if (atomicRead(&status[dst]) == undecided) {
// and is undecided -> include
atomicWrite(&status[dst], included);
}
}
if (status[src] == undecided || status[dst] == undecided) { // if either is still undecided, keep it in WL
wl2[criticalAdd(&wl2size, 1)] = e;
}
}
}

static double OMPmis_edge(const ECLgraph& g, const int* const sp, data_type* const priority, unsigned char* const status)
{
bool* lost = new bool [g.nodes];
const int size = std::max(g.edges, g.nodes);
int* wl1 = new int [size];
int* wl2 = new int [size];
int wl1size;
int wl2size;

init(g, sp, priority, status, lost, wl1, wl1size);

timeval beg, end;
gettimeofday(&beg, NULL);

int iter = 0;
do {
iter++;
wl2size = 0;

// edge pass
mis(g, sp, priority, status, lost, wl1, wl1size, wl2, wl2size);


// vertex pass
mis_vertex_pass(g, sp, priority, status, lost, wl1, wl1size, wl2, wl2size);


std::fill(lost, lost + g.nodes, false);
std::swap(wl1, wl2);
wl1size = wl2size;
} while (wl1size > 0);
// include all remaining nodes that have no edges
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < g.nodes; i++) {
if (status[i] == undecided)
status[i] = included;
}

gettimeofday(&end, NULL);
const double runtime = end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;

// determine and print set size
int cnt = 0;
for (int v = 0; v < g.nodes; v++) {
if (status[v] == included) cnt++;
}
printf("iterations: %d,  elements in set: %d (%.1f%%)\n", iter, cnt, 100.0 * cnt / g.nodes);

delete [] lost;
delete [] wl1;
delete [] wl2;
return runtime;
}
