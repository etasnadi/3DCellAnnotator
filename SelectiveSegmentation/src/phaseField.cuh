#ifndef _PHASEFIELD_CUH
#define _PHASEFIELD_CUH 1

#include "common.cuh"

#define _SURF_PT_MARK 1
#define _NOT_SURF_PT_MARK 0

using namespace commonns;

void regularisePhaseField(float* levelSet, GridParams gProps, AlgParams aParams);

__global__ void createIdFunc(Func3D<int3> func);
__global__ void markSurfacePoints(Func3D<float> levelSet, Func3D<uint32_t> result, GridParams gParams);

typedef struct gt {
	int than;
	__host__ __device__ gt(int a_than) : than(a_than) {}
	__host__ __device__
	bool operator()(const int x){
		return x > than;
	}
} gt;

std::pair<int3, int3> getFieldExtrema(dchunk_float& d_levelSet, GridParams gridProps);
dchunk_float::uptr_t makeFrame(dchunk_float& d_field, Size3D size, int frameSize);

#endif
