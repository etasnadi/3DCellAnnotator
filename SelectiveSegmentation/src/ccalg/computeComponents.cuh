#pragma once

#include <memory>
#include <iostream>
#include "../cudatools/deviceChunk.cuh"
#include "../cudatools/cudatools.cuh"
#include "../cudatools/types.cuh"

#define INACTIVE 0
#define ACTIVE 1

#define INVALID_NODE -1

#define CC_TRUE 1
#define CC_FALSE 0

#define ALL_CONTRACTED allContracted[0]
#define ALL_INACTIVE allInactive[0]

typedef struct {
	int numOfSMs;
	int numOfCoresPerSM;

public:
	int getNumOfCores() {
		return numOfSMs*numOfCoresPerSM;
	}
	void print() {
		printf("Device info:\n\tnumber of cores: %d\n\tnumberOfSMs: %d\n\tnumberOfCoresPerSM: %d\n", getNumOfCores(), numOfSMs, numOfCoresPerSM);
	}
} device_spec;

typedef struct {
	int numOfBlocks;
	int numOfThreadsPerBlock;
	int getNumOfThreads() {
		return numOfBlocks*numOfThreadsPerBlock;
	}
} problem_spec;

typedef struct {
	int batchSizeNodes;
	int batchSizeEdges;
public:
	void print(){
		//printf("Batch info: (batchSizeNodes, batchSizeEdges): (%d,%d)\n", batchSizeNodes, batchSizeEdges);
	}
} batch_info;

typedef struct {
	int PJIterations;
	int hookIterations;
public:
	void printInfo(){
		printf("Status: (hooks, pointerJumps): (%d,%d)\n", hookIterations, PJIterations);
	}
} alg_stat;

int calcBatchSize(device_spec device, int problemSize);
unique_ptr<DevIntChk> computeCC(std::unique_ptr<DevIntPairChk>& edges, int numOfNodes);
int makeDense(unique_ptr<DevIntChk>& components, int numOfNodes);
