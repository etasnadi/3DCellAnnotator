#include <iostream>

#include <thrust/transform.h>
#include <thrust/device_ptr.h>

#include <boost/log/trivial.hpp>

#include "cudatools/deviceChunk.cuh"
#include "phaseField.cuh"
#include "evolve.cuh"
#include "cudatools/function.cuh"
#include "marchingCubes.cuh"

using namespace commonns;
using namespace std;
using namespace thrust;

__global__ void regularisationKernel(float* phase, float* Lphase, float* LLphase, Size3D gridDim, AlgParams aParams){
	Point3D p = getThread3D();

	FltFunc3D phasef(gridDim, phase);
	FltFunc3D Lphasef(gridDim, Lphase);
	FltFunc3D LLphasef(gridDim, LLphase);

	if(p >= 0 && p < Point3D(gridDim)){
		float wsq = pow(aParams.w,2);
		float norm = aParams.regNormTerm;
		phasef[p] = phasef[p] + norm*(-1*(wsq/16) * LLphasef[p] - Lphasef[p] - (21/wsq)*(pow(phasef[p],3)-phasef[p]));
	}
}

// Parameters needed:
// w, regNormTerm
// gridSize, gridRes

void regularisePhaseField(float* phase, GridParams gProps, AlgParams aParams){
	Size3D gridDim = gProps.gridSize;
	int gridRes = gProps.gridRes;
	int arrLen = gridDim.vol();
	GpuConf3D conf(gridDim, SQ_TPB_3D);

	DeviceChunk<float> Lphase(arrLen);
	DeviceChunk<float> LLphase(arrLen);

	computeL<<<conf.grid(), conf.block()>>>(Lphase.getPtr(), phase, gridDim, gridRes);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	computeL<<<conf.grid(), conf.block()>>>(LLphase.getPtr(), Lphase.getPtr(), gridDim, gridRes);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	regularisationKernel<<<conf.grid(), conf.block()>>>(phase, Lphase.getPtr(), LLphase.getPtr(), gridDim, aParams);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

// Surface points scanning
__global__ void markSurfacePoints(Func3D<float> levelSet, Func3D<uint32_t> result, GridParams gParams){
	int3 p = getThread3D().getInt3();
	int3 size = gParams.gridSize.geti3();
	const uint32_t SURF_PT(_SURF_PT_MARK);
	const uint32_t NOT_SURF_PT(_NOT_SURF_PT_MARK);
	if(p < size){
		result[p] = NOT_SURF_PT;
		Cubes marchingCubes(levelSet);
		if(p < size-IONES3 && marchingCubes.onIsoSurface(p)){
			result[p] = SURF_PT;
		}
	}
}

__global__ void createIdFunc(Func3D<int3> func){
	int3 p = getThread3D().getInt3();
	int3 size = func.getSize().geti3();
	if(p < size){
		func[p] = p;
	}
}

dchunk<int3>::uptr_t getSurfacePointsList(dchunk_float& d_levelSet, GridParams gParams){
	Size3D gridSize = gParams.gridSize;

	func3_f levelSet = d_levelSet.funcView(gridSize);
	GpuConf confGrid = WorkManager(gridSize, 4).conf();
	dchunk_uint32 d_surfaceMask(gridSize.vol());
	d_surfaceMask.fill(uint32_t(_NOT_SURF_PT_MARK));


	func3<uint32_t> surfaceMask = d_surfaceMask.funcView(gridSize);
	markSurfacePoints<<<confGrid.grid(), confGrid.block()>>>(levelSet, surfaceMask, gParams);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	dchunk<int3> d_idFunc(gridSize.vol());
	func3<int3> idFunc = d_idFunc.funcView(gridSize);
	createIdFunc<<<confGrid.grid(), confGrid.block()>>>(idFunc);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	int nSurfPts = count(device, d_surfaceMask.tbegin(), d_surfaceMask.tend(), _SURF_PT_MARK);
	dchunk<int3>::uptr_t d_surfPts = dchunk<int3>::make_uptr(nSurfPts);

	copy_if(device, d_idFunc.tbegin(), d_idFunc.tend(), d_surfaceMask.tbegin(), d_surfPts->tbegin(), gt(_NOT_SURF_PT_MARK));

	return d_surfPts;
}

// Level set clipping

struct compare_pts {

	int3 mask;

	__host__ __device__ compare_pts(int3 a_mask) : mask(a_mask){}

	__host__ __device__ bool operator()(int3 a, int3 b){
		return a*mask < b*mask;
	}
};

std::pair<int3, int3> getFieldExtrema(dchunk_float& d_levelSet, GridParams gridProps){
	std::cout << "getFieldExtrema" << gridProps.gridSize << std::endl;
	dchunk<int3>::uptr_t surfPts = getSurfacePointsList(d_levelSet, gridProps);

	auto axis_X = thrust::minmax_element(
			thrust::device,
			surfPts->tbegin(),
			surfPts->tend(),
			compare_pts(EU.getInt3()));

	auto axis_Y = thrust::minmax_element(
			thrust::device,
			surfPts->tbegin(),
			surfPts->tend(),
			compare_pts(EV.getInt3()));

	auto axis_Z = thrust::minmax_element(
			thrust::device,
			surfPts->tbegin(),
			surfPts->tend(),
			compare_pts(EW.getInt3()));

	 int3 minX = *axis_X.first;
	 int3 maxX = *axis_X.second+1;

	 int3 minY = *axis_Y.first;
	 int3 maxY = *axis_Y.second+1;

	 int3 minZ = *axis_Z.first;
	 int3 maxZ = *axis_Z.second+1;

	 int3 mins = make_int3(minX.x, minY.y, minZ.z);
	 int3 maxs = make_int3(maxX.x, maxY.y, maxZ.z);

	return std::make_pair(mins, maxs);
}

// input < output+2!
__global__ void copyIntoFrame(func3_f input, func3_f output, int frameSize){
	int3 p = getThread3D().getInt3();
	int3 sizei = input.getSize().geti3();
	int3 sizeo = output.getSize().geti3();
	if(p >= IZEROS3 && p < sizei){
		output[p+frameSize*IONES3] = input[p];
	}
}

dchunk_float::uptr_t makeFrame(dchunk_float& d_field, Size3D size, int frameSize){
	func3_f f_field = d_field.funcView(size);

	int3 resultSize = size.geti3()+ 2*frameSize*IONES3;
	dchunk_float::uptr_t ud_result = dchunk_float::make_uptr(mul(resultSize));
	ud_result->fill(0.0f);
	func3_f f_result = ud_result->funcView(Size3D(resultSize));

	GpuConf conf = WorkManager(size, 4).conf();
	copyIntoFrame<<<conf.grid(), conf.block()>>>(f_field, f_result, frameSize);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	return ud_result;
}
