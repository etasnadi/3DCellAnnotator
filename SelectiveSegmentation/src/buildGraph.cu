#include <iostream>
#include <functional>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include "common.cuh"
#include "cudatools/errorHandling.cuh"
#include "buildGraph.cuh"
#include "cudatools/deviceChunk.cuh"
#include "marchingCubes.cuh"
using namespace std;

__device__ int calcNNeighs(float *d_data, Point3D p, Size3D size){
	Point3D es[3] = {EU, EV, EW};
	int count = 0;
	Func3D<float> levelSet(size, d_data);
	Cubes marchingCubes(levelSet);
	if(p >= 0 && p < Point3D(size)-1){
		if(marchingCubes.onIsoSurface(p.getInt3())){
			count = 0;
			for(int neigh_id = 0; neigh_id < 3; neigh_id++){
				Point3D p_ne = p+es[neigh_id];
				if(p_ne >=0 && p_ne < Point3D(size)-1 && marchingCubes.onIsoSurface(p_ne.getInt3())){
					++count;// Apply this. Input: p, p_ne
				}
			}
		}
	}
	return count;
}

__global__ void computeNNeighs(int *d_neighbors, float *d_data, Size3D size){
	initSharedMemory();
	Point3D p = getThread3D();
	IntFunc3D neigh(size, d_neighbors);
	if(p>=0 && p < Point3D(size)-1){
		neigh[p] = calcNNeighs(d_data, p, size);
	} else if( ((p*1 != ZEROS3) || !(p-Point3D(size)-1 > 0)) && p < size) {
		neigh[p] = 0;
	}
}

__global__ void computeGraph(float *d_data, IntPair* d_graph, int *d_neighbors, Size3D size){
	initSharedMemory();
	// Should supply by the lambda
	Point3D p = getThread3D();
	IntFunc3D neigh(size, d_neighbors);
	Func3D<float> levelSet(size, d_data);
	Cubes marchingCubes(levelSet);
	Point3D es[3] = {EU, EV, EW};
	int count;
	if(p >= 0 && p < Point3D(size)-1){
		if(marchingCubes.onIsoSurface(p.getInt3())){
			count = 0;
			for(int neig_id = 0; neig_id < 3; neig_id++){
				Point3D p_ne = p+es[neig_id];
				if(p_ne >= 0 && p_ne < size && marchingCubes.onIsoSurface(p_ne.getInt3())){
					Point3D tran = getlinTrans(size); // Apply these. Input: p, p_ne
					IntPair edge = make_int2(p*tran, p_ne*tran);
					int edgeId = neigh[p]+count;
					d_graph[edgeId] = edge;
					count++;
				}
			}
		}		
	}
}

void launchBuildKernel(float *d_data, IntPair *d_graph, int *d_neighbors, Size3D size){
	GpuConf3D conf(size, 4);
	computeGraph<<<conf.grid(), conf.block(), MC_LOC_SIZE*64>>>(d_data, d_graph, d_neighbors, size);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

void fillNeighborsArray(DevFloatChk& data, DevIntChk& neighs, Size3D dims){
	GpuConf3D conf(dims, 4);
	computeNNeighs<<<conf.grid(), conf.block(), MC_LOC_SIZE*64>>>(neighs.getPtr(), data.getPtr(), dims);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	thrust::exclusive_scan(neighs.getTPtr(), neighs.getTPtr() + neighs.getElements(), neighs.getTPtr());
}

// buffer: the storage for the graph
unique_ptr<DevIntPairChk> buildGraph(DevFloatChk& data, DevIntChk& neighs, Size3D size){
	int numOfEdges = neighs.getVal(neighs.getElements()-1);
	auto ccGraph = unique_ptr<DevIntPairChk>(new DevIntPairChk(numOfEdges));
	launchBuildKernel(data.getPtr(), ccGraph.get()->getPtr(), neighs.getPtr(), size);
	return ccGraph;
}
