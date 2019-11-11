#ifndef BUILDGRAPH_CUH
#define BUILDGRAPH_CUH 1

#include "common.cuh"

using namespace std;

void fillNeighborsArray(DevFloatChk& data, DevIntChk& neighs, Size3D dims);
unique_ptr<DevIntPairChk> buildGraph(DevFloatChk& data, DevIntChk& neighs, Size3D size);

#endif
