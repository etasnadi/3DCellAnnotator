#include <stdio.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/uninitialized_fill.h>
#include <thrust/device_malloc.h>

#include "../cudatools/types.cuh"
#include "computeComponents.cuh"

// Aux functions.

__device__ int _max(int a, int b) {
	if (a > b) return a;
	else return b;
}

__device__ int _min(int a, int b) {
	if (a < b) return a;
	else return b;
}

/*
For performance reasons, the numOfBlocks should be equal to the number of SMs in the hardware, and the threadsPerBlock should be equal to the number of single precision cores
in a SM.
*/
int calcBatchSize(device_spec device, int problemSize) {
	int numOfThreads = device.numOfSMs*device.numOfCoresPerSM;
	int batchSize;

	batchSize = problemSize / numOfThreads;

	if (problemSize % numOfThreads != 0) {
		batchSize++;
	}
	return batchSize;
}

__global__ void hookNodes(int batchSize, int numOfEdges, int numOfNodes, IntPair* edges, int* edgeStatus, int* parent) {
	int batchOffset = batchSize*getThread();
	//printf("hookNodes called. batchOffset=%d\n", batchOffset);

	for (int edgeId = batchOffset; edgeId < batchOffset + batchSize; edgeId++) {
		if (edgeId < numOfEdges) {
			int u = edges[edgeId].x;
			int v = edges[edgeId].y;

			if (edgeStatus[edgeId] == ACTIVE && parent[u] != parent[v]) {
				//printf("Hooking %d %d | parent[%d]=parent[%d]\n", u, v, parent[u], parent[v]);
				int mx = _max(parent[u], parent[v]);
				int mn = _min(parent[u], parent[v]);
				int it = 0;
				if (it % 2 == 0) {
					parent[mn] = mx;
				}
				else {
					parent[mx] = mn;
				}
			}
		}
	}
}

//Contraction alg.

/**
Jumps the parent pointer one level upwards.
ALL_CONTRACTED meaning: if all of the trees are stars. Formally:
- there is exactly one node at the level 0 - called root node <=> the fixed point of the parent relation (parent(x)=x)
- the other nodes are at level 1 - called leaf nodes <=> for all node x in the set of the leaf nodes: parent(parent(x))=parent(x)
*/

__global__ void pointerJump(int batchSize, int numOfNodes, int* parentSrc, int* parentDst, int* allContracted) {
	int batchOffset = batchSize*getThread();
	for (int nodeId = batchOffset; nodeId < batchOffset + batchSize; nodeId++) {
		if (nodeId < numOfNodes && parentSrc[nodeId] != INVALID_NODE) {
			if (parentSrc[parentSrc[nodeId]] != parentSrc[nodeId]) {
				if (ALL_CONTRACTED == CC_TRUE) {
					ALL_CONTRACTED = CC_FALSE;
				}
				parentDst[nodeId] = parentSrc[parentSrc[nodeId]];
			}
			else {
				parentDst[nodeId] = parentSrc[nodeId];
			}
		}
	}
}

/*
Edge hiding operation.

The kernel scans the edge list [(u,v)], and examines the parents of the endpoints: p(u) and p(v).
It can be proven that the hook operation never cuts a component, so if p(u) equals p(v), we do not need to care
about the edge (u,v) in the future so we make it inactive.
If (u,v) is ACTIVE and p(u)==p(v) then mark (u,v) INACTIVE
If (u,v) is ACTIVE and p(u)!=p(V) then switch a global variable ON (we should continue the hooking process)
If (u,v) is INACTIVE then do nothing. Because if an edge is INACTIVE it implies that u and v are in the same component
because we make an edge INACTIVE when u and v are in the same component.
*/

__global__ void hideEdges(int batchSize, int numOfEdges, int* edgeStatus, int* parent, int* allInactive, IntPair* edges) {
	int batchOffset = batchSize*getThread();
	//printf("hideEdges called. batchOffset=%d\n", batchOffset);
	for (int edgeId = batchOffset; edgeId < batchOffset + batchSize; edgeId++) {
		if (edgeId < numOfEdges) {
			//printf("edgeId=%d status=%d u=%d v=%d p(u)=%d p(v)=%d\n", edgeId, edgeStatus[edgeId], edges[EDGE_FROM], edges[EDGE_TO], parent[edges[EDGE_FROM]], parent[edges[EDGE_TO]]);
			if (edgeStatus[edgeId] == ACTIVE &&  parent[edges[edgeId].x] == parent[edges[edgeId].y]) {
				edgeStatus[edgeId] = INACTIVE;
			}
			else if (edgeStatus[edgeId] == ACTIVE && parent[edges[edgeId].x] != parent[edges[edgeId].y]) {
				if (ALL_INACTIVE == CC_TRUE) {
					ALL_INACTIVE = CC_FALSE;
				}
			}
		}
	}
}

/*
Initialises the parent relations: initially all of the possible nodes (that are not neccessarily nodes because of the sparse storage) to INVALID.
Example: if we have 10 nodes, but we actually use only the 4 and 6, then we should set the parents of all of the nodes to invalid, and in an another kernel call
we should set the parents of 4 and 6 to themselves. We should care about the invalid nodes to not to execute pointer-referencing trhrough them for example in the
pointer jumping kernel.
*/
__global__ void initVertexIds(int batchSize, int numOfNodes, int* parentA, int* parentB) {
	int batchOffset = batchSize*getThread();
	//printf("initVertexIds called. batchOffset=%d\n", batchOffset);
	for (int nodeId = batchOffset; nodeId < batchOffset + batchSize; nodeId++) {
		if (nodeId < numOfNodes) {
			parentA[nodeId] = INVALID_NODE;
			parentB[nodeId] = INVALID_NODE;
		}
	}
}

/*
Initialises the edge status list and the each of the parent relations.
- the edge status array will be filled with ACTIVE-s.
- every parent of every node found in the edge list will be itself.
*/
__global__ void initEdgeStatus(int batchSize, int numOfEdges, int* edgeStatus, IntPair* edges, int* parentA, int* parentB) {
	int batchOffset = batchSize*getThread();
	//printf("Init edge status called. batchOffset=%d\n", batchOffset);
	for (int edgeId = batchOffset; edgeId < batchOffset + batchSize; edgeId++) {
		if (edgeId < numOfEdges) {
			edgeStatus[edgeId] = ACTIVE;
			int u = edges[edgeId].x;
			int v = edges[edgeId].y;
			parentA[u] = u;
			parentA[v] = v;
			parentB[u] = u;
			parentB[v] = v;
		}
	}
}

// Component id contraction part for the make dense function.

__global__ void createComponentIdsMap(int* result, int* componentIds, int numberOfComponentIds) {
	int idx = getThread();
	if (idx < numberOfComponentIds) {
		if (componentIds[idx] != -1) {
			if (componentIds[0] == -1) { // If the 0th component is the component with id -1 then...
				result[componentIds[idx]] = idx - 1;
			}
			else {
				result[componentIds[idx]] = idx;
			}
		}
	}
}

__global__ void remapComponentIds(int* componentIds, int* map, int numOfNodes) {
	int idx = getThread();
	if (idx < numOfNodes) {
		if (componentIds[idx] != -1) {
			componentIds[idx] = map[componentIds[idx]];
		}
	}
}

// Contracts the component ids on the GPU. It needs an additional (numOfNodes+numberOfComponents)*sizeof(int) memory.
int makeDense(unique_ptr<DevIntChk>& components, int numOfNodes) {
	DevIntChk componentIds(numOfNodes);

	gpuErrchk(cudaMemcpy(componentIds.getPtr(), components.get()->getPtr(), numOfNodes * sizeof(int), cudaMemcpyDeviceToDevice));
	thrust::device_ptr<int> componentIds_ptr(componentIds.getPtr());
	thrust::sort(componentIds_ptr, componentIds_ptr + numOfNodes);
	thrust::device_ptr<int> lastElement_ptr = thrust::unique(componentIds_ptr, componentIds_ptr + numOfNodes);


	//	Now the componentIds array are sorted, there are unique elements with ids from -1...
	int numberOfComponentIds = (lastElement_ptr - componentIds_ptr);
	//printf("Size: %d\n", numberOfComponentIds);

	// Move the unique ordered component ids in an other array and reuse the componentIds array as a map
	DevIntChk uniqueComponentIds(numberOfComponentIds);
	gpuErrchk(cudaMemcpy(uniqueComponentIds.getPtr(), componentIds.getPtr(), numberOfComponentIds * sizeof(int), cudaMemcpyDeviceToDevice));

	// Fill the map with default invalid values (-2)
	thrust::fill(componentIds_ptr, componentIds_ptr + numOfNodes, -2);

	// Fill the map with the component id mapping
	int blockSize = 64; // 64 threads / block
	int numOfBlocks = numberOfComponentIds / blockSize + 1;
	createComponentIdsMap << <numOfBlocks, blockSize >> >(componentIds.getPtr(), uniqueComponentIds.getPtr(), numberOfComponentIds);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());


	numOfBlocks = numOfNodes / blockSize + 1;
	remapComponentIds << <numOfBlocks, blockSize >> >(components.get()->getPtr(), componentIds.getPtr(), numOfNodes);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	return numberOfComponentIds;
}

unique_ptr<DevIntChk> computeCC(std::unique_ptr<DevIntPairChk>& edges, int numOfNodes) {
	device_spec device;
	device.numOfSMs = 15;
	device.numOfCoresPerSM = 32;

	problem_spec problem;
	problem.numOfBlocks = device.numOfSMs;
	problem.numOfThreadsPerBlock = device.numOfCoresPerSM;

	int numOfEdges = edges.get()->getElements();
	batch_info batch;
	batch.batchSizeNodes = calcBatchSize(device, numOfNodes);
	batch.batchSizeEdges = calcBatchSize(device, numOfEdges);

	DevIntChk edgeStatus(numOfEdges);
	DevIntChk isAllInact(1), isAllContr(1);

	auto parentA = DevIntChk::make_uptr(numOfNodes);
	auto parentB = DevIntChk::make_uptr(numOfNodes);

	// Step 0: Initialisation step
	initVertexIds << <problem.numOfBlocks, problem.numOfThreadsPerBlock >> >(
		batch.batchSizeNodes,
		numOfNodes,
		parentA.get()->getPtr(),
		parentB.get()->getPtr());
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	initEdgeStatus << <problem.numOfBlocks, problem.numOfThreadsPerBlock >> >(
		batch.batchSizeEdges,
		numOfEdges,
		edgeStatus.getPtr(),
		edges.get()->getPtr(),
		parentA.get()->getPtr(),
		parentB.get()->getPtr());
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	alg_stat stat;
	stat.hookIterations = 0;
	do {
		stat.PJIterations = 0;
		// Step 1: Hooking
		hookNodes << <problem.numOfBlocks, problem.numOfThreadsPerBlock >> >(
			batch.batchSizeEdges,
			numOfEdges,
			numOfNodes,
			edges.get()->getPtr(),
			edgeStatus.getPtr(),
			parentA.get()->getPtr());
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		// Step2: Full pointer jump
		do {
			// Optimistic way: set up as the contraction is happened for each tree, transfer the info to the GPU and let it change if it is not true.
			isAllContr.putVal(CC_TRUE);

			// Jump the pointers. In even iterations: A->B, in the odds: B->A
			// in even iterations...

			pointerJump << <problem.numOfBlocks, problem.numOfThreadsPerBlock >> >(
				batch.batchSizeNodes,
				numOfNodes,
				parentA.get()->getPtr(),
				parentB.get()->getPtr(),
				isAllContr.getPtr());
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());

			parentA.swap(parentB);

			//Check whether our first assumption was true or not. If not, we should continue the pointer jumping.

			stat.PJIterations++;
		} while (isAllContr.getVal() == CC_FALSE);
		// Step3: Edge hiding

		// During edge hiding (edge inactivation we count the edges that cannot be inactivated).
		isAllInact.putVal(CC_TRUE);

		hideEdges << <problem.numOfBlocks, problem.numOfThreadsPerBlock >> >(
			batch.batchSizeEdges,
			numOfEdges,
			edgeStatus.getPtr(),
			parentA.get()->getPtr(),
			isAllInact.getPtr(),
			edges.get()->getPtr());
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		stat.hookIterations++;

	} while (isAllInact.getVal() == CC_FALSE);

	return parentA;
}
