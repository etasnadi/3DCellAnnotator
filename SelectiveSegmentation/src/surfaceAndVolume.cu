#include <iostream>
#include <stdio.h>
#include <math.h>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <boost/log/trivial.hpp>

#include "cudatools/errorHandling.cuh"
#include "surfaceAndVolume.cuh"
#include "marchingCubes.cuh"
#include "common.cuh"

using namespace std;
using namespace commonns;
using namespace thrust;

#define COMPUTE_ADVANCED_STATS 1

// Utility

float calculateSum(float *d_data, int size){
	thrust::device_ptr<float> dptr_d_data(d_data);
	return thrust::reduce(dptr_d_data, dptr_d_data+size, (float) 0, thrust::plus<float>());
}

float4 calculateFloat4Sum(float4 *d_data, int size){
	thrust::device_ptr<float4> dptr_d_data(d_data);
	return thrust::reduce(dptr_d_data, dptr_d_data+size, make_float4(0,0,0,0), thrust::plus<float4>());
}

float3 calculateFloat3Sum(float3 *d_data, int size){
	thrust::device_ptr<float3> dptr_d_data(d_data);
	return thrust::reduce(dptr_d_data, dptr_d_data+size, make_float3(0.0f, 0.0f, 0.0f), thrust::plus<float3>());
}

// Surface

__device__ float calculateSurfaceInCube(float *d_data, Point3D p, Size3D size){
	float voxelSurface(0.0f);
	Triangle triangles[5];

	Func3D<float> levelSet(size, d_data);
	Cubes marchingCubes(levelSet);

	if(marchingCubes.onIsoSurface(p.getInt3())){
    	int nOfTriangles = marchingCubes.getVoxelTriangles(p.getInt3(), triangles);
    	for(uint8_t triId = 0; triId < nOfTriangles; triId++){
    		voxelSurface += triangles[int(triId)].area();
    	}
    }
	return voxelSurface;
}

__global__ void computeSurfaceKernel(float *d_data, float *d_surface, float dh, Size3D size){
	// Should allocate the shared memory to the marching cubes!
	initSharedMemory();

	Point3D p = getThread3D();
	FltFunc3D surf(size, d_surface);
	if(p >=0 && p < Point3D(size)-1){
		surf[p] = pow(dh,2)*calculateSurfaceInCube(d_data, p, size);
	} else if(!(p-Point3D(size)-1 > 0) && p < Point3D(size)) {
		surf[p] = 0;
	}
}

void computeSurface(DevFloatChk& data, DevFloatChk& surface, float dh, Size3D size){
	GpuConf3D conf(size, 4);
	computeSurfaceKernel<<<conf.grid(), conf.block(), MC_LOC_SIZE*64>>>(data.getPtr(), surface.getPtr(), dh, size);
	KERNEL_ERROR_CHECK("calculateSurfaceKernel error");
}

float getSurface_global(DevFloatChk& data, DevFloatChk& surface, float dh, Size3D size){
	GpuConf3D conf(size, 4);
	computeSurfaceKernel<<<conf.grid(), conf.block(), MC_LOC_SIZE*64>>>(data.getPtr(), surface.getPtr(), dh, size);
	KERNEL_ERROR_CHECK("calculateSurfaceKernel error");
	return calculateSum(surface.getPtr(), size.vol());
}

// Volume

__device__ int checkOrientation(float3 A, float3 B){
	if(A*B >= 0){
		return 1;
	} else {
		return -1;
	}
}

__device__ float computeVolume(float *d_data, Triangle tri, Point3D p, Size3D size){
	float3 refVector = p.getFloat3()-tri.A;
	FltFunc3D data(size, d_data);
	// determine that the reference point (i,j,k) is below or above the surface
	int refSign = -1;
	if(data[p] >= 0){
		refSign=1;
	}
	refSign = sgn(data[p]);
	
	// create vectors corresponding to sides of the triangle
	float3 bma = tri.B-tri.A;
	float3 cma = tri.C-tri.A;
	
	// determine the orientation of s1 x s2 with respect to the surface
	float3 cr = cross(bma,cma);
	int orientation = checkOrientation(cr, refVector);
	float V = (cr*tri.A)/6;
	if(refSign == 1){
		if(orientation == 1){
			return V;
		} else {
			return -V;
		}
	} else {
		if(orientation == 1){
			return -V;
		} else {
			return V;
		}
	}
}

__device__ float calculateVolumeForCube(float *d_data, Point3D p, Size3D size){
	float volume(0);

    Triangle tris[5];
    Func3D<float> levelSet(size, d_data);
    Cubes marchingCubes(levelSet);
    int nTr = marchingCubes.getVoxelTriangles(p.getInt3(), tris);

    for(uint8_t triId = 0; triId < nTr; triId++){
    		Triangle tr = tris[triId];
			volume += computeVolume(d_data,tr,Point3D(int(tr.A.x),int(tr.A.y),int(tr.A.z)),size);
    }

	return volume;
}

__global__ void calculateVolumeKernel(float *d_data, float *d_volume, float dh, Size3D size){
	initSharedMemory();
	Point3D p = getThread3D();
	FltFunc3D vol(size, d_volume);
	if(p >= 0 && p < Point3D(size)-1){
		vol[p] = pow(dh, 3)*calculateVolumeForCube(d_data, p, size);
	} else if((p*1 != p  || !(p-Point3D(size)-1 > 0)) && p < Point3D(size)){
		vol[p] = 0;
	}
}

void launchCalculateVolume(DevFloatChk& data, DevFloatChk& volume, float dh, Size3D size){
	GpuConf3D conf(size, 4);
	calculateVolumeKernel<<<conf.grid(), conf.block(), MC_LOC_SIZE*64>>>(data.getPtr(), volume.getPtr(), dh, size);
	KERNEL_ERROR_CHECK("calculateVolumeKernel error");
}

float getVolume_global(DevFloatChk& data, DevFloatChk& volume, float dh, Size3D size){
	GpuConf3D conf(size, 4);
	calculateVolumeKernel<<<conf.grid(), conf.block(), MC_LOC_SIZE*64>>>(data.getPtr(), volume.getPtr(), dh, size);
	KERNEL_ERROR_CHECK("calculateVolumeKernel error");
	return calculateSum(volume.getPtr(), size.vol());
}

// Center of gravity

__device__ float3 cogContribVoxel(float *d_data, float dh, Point3D p, Size3D size){
	float3 preCOG = make_float3(0.0f, 0.0f, 0.0f);
	Func3D<float> levelSet(size, d_data);
	Cubes marchingCubes(levelSet);
	Triangle triangles[5];
	int nOfTriangles = marchingCubes.getVoxelTriangles(p.getInt3(), triangles);
	for(int i = 0; i < nOfTriangles; i++){
		Triangle tri = triangles[i];
		float3 cogArea = tri.cog() * tri.area();
		preCOG = preCOG + cogArea;
	}

	return dh*preCOG;
}

__global__ void centerOfGravityKernel(float *d_data, float3 *d_centerOfGravity, float dh, Size3D size){
	initSharedMemory();
	Point3D p = getThread3D();
	Func3D<float3> cog(size, d_centerOfGravity);
	if(p >= 0 && p < Point3D(size)-1){
		cog[p] = cogContribVoxel(d_data, dh, p, size);
	}
}

void launchCenterOfGravity(float *d_data, float3 *d_centerOfGravity, float dh, Size3D size){
	GpuConf3D conf(size, 4);
	centerOfGravityKernel<<<conf.grid(), conf.block(), MC_LOC_SIZE*64>>>(d_data, d_centerOfGravity, dh, size);
	KERNEL_ERROR_CHECK("centerOfGravityKernel error");
}

// Global center of gravity...

float3 getCOG(DevFloatChk& data, DeviceChunk<float3>& centerOfGravity, float dh, Size3D gridDims){
	launchCenterOfGravity(data.getPtr(), centerOfGravity.getPtr(), dh, gridDims);
	float3 preCOG = calculateFloat3Sum(centerOfGravity.getPtr(), gridDims.vol());
	return preCOG;
}

// Second moment

__device__ float cubeContribution(float* d_data, float3 cogObj, float dh, Point3D p, Size3D size){
	if(!((Point3D(size)-1-p)>0)){
		return 0;
	}

	float cogContribution(0.0f);

	Triangle triangles[5];
	Func3D<float> levelSet(size, d_data);
	Cubes marchingCubes(levelSet);

    if(marchingCubes.onIsoSurface(p.getInt3())){
		int nOfTriangles = marchingCubes.getVoxelTriangles(p.getInt3(), triangles);
		for(uint8_t triId = 0; triId < nOfTriangles; triId++){
			Triangle tri = triangles[int(triId)];
			float3 cogTri = tri.cog();
			float3 R = cogTri-cogObj;
			float Rsq = R*R;
			float triContrib = Rsq*tri.area();
			cogContribution += triContrib;
		}
	}
	
	return cogContribution;
}

__global__ void secondMomentKernel(float *d_levelSet, float *d_secondMoment, float3* d_cog, int *d_nodeID, float dh, Size3D size){
	initSharedMemory();
	Point3D p = getThread3D();
	IntFunc3D f_nodeID(size, d_nodeID);
	FltFunc3D f_secondMoment(size, d_secondMoment);
	if(p>=0 && p < Point3D(size)) {
		if(p < Point3D(size)-1) {
			int nodeID = f_nodeID[p];
			if(nodeID == -1) {
				f_secondMoment[p] = 0;
			} else {
				float3 cog = d_cog[nodeID+1];
				f_secondMoment[p] = cubeContribution(d_levelSet, cog, dh, p, size);
			}
		} else {
			f_secondMoment[p] = 0;
		}
	}
}

float getSecondMoment_global(DevFloatChk& levelSet, DevFloatChk& secondMoment, DeviceChunk<float3>& cog, DevIntChk& nodeID, float dh, Size3D gridDims){
	GpuConf3D conf(gridDims, 4);
	secondMomentKernel<<<conf.grid(), conf.block(), MC_LOC_SIZE*64>>>(levelSet.getPtr(), secondMoment.getPtr(), cog.getPtr(), nodeID.getPtr(), dh, gridDims);
	KERNEL_ERROR_CHECK("secondMomentKernel error");
	return calculateSum(secondMoment.getPtr(), gridDims.vol());
}

CompStats extractComponentStatistics(float* d_statistics, DevIntChk& ccresult, int numOfComponents, commonns::Dimensions dims){
	// Step 0.: copy the ccresult array to a brand new working array to do not mess it up the further steps of the segmentation.

	int ccNelements = ccresult.getElements();

	DevIntChk cCcresult(ccNelements);
	ccresult.copy(cCcresult);
	
	// Step 1.: Sort the ccresult_copy and the statistics array by considering it as ccresult->statistics map and sort it by considering the ccresult as the key
	thrust::device_ptr<int> ccresult_copy_ptr = cCcresult.getTPtr();
	thrust::device_ptr<float> statistics_ptr(d_statistics);
	thrust::sort_by_key(ccresult_copy_ptr, ccresult_copy_ptr + ccNelements, statistics_ptr);
	
	// Step 2.: Parallel reduce the values assigned to the same key to the output arrays
	auto resultComponentIds = unique_ptr<DevIntChk>(new DevIntChk(ccNelements));
	auto resultComponentContributions = unique_ptr<DevFloatChk>(new DevFloatChk(ccNelements));

	thrust::reduce_by_key(
		ccresult_copy_ptr, ccresult_copy_ptr + dims.s(), 
		statistics_ptr,
		
		resultComponentIds.get()->getTPtr(),
		resultComponentContributions.get()->getTPtr());

	CompStats ret;
	ret.first = std::move(resultComponentIds);
	ret.second = std::move(resultComponentContributions);
	return ret;
}

CompStatsF3 extractComponentStatistics(float3* d_statistics, DevIntChk& ccresult, int numOfComponents, commonns::Dimensions dims){
	// Step 0.: copy the ccresult array to a brand new working array to do not mess it up the further steps of the segmentation.
	int ccNelements = ccresult.getElements();

	DevIntChk cCcresult(ccNelements);
	ccresult.copy(cCcresult);
	
	// Step 1.: Sort the ccresult_copy and the statistics array by considering it as ccresult->statistics map and sort it by considering the ccresult as the key
	thrust::device_ptr<int> ccresult_copy_ptr = cCcresult.getTPtr();
	thrust::device_ptr<float3> statistics_ptr(d_statistics);
	thrust::sort_by_key(ccresult_copy_ptr, ccresult_copy_ptr + ccNelements, statistics_ptr);
	
	// Step 2.: Parallel reduce the values assigned to the same key to the output arrays
	auto resultComponentIds = unique_ptr<DevIntChk>(new DevIntChk(ccNelements));
	auto resultComponentContributions = unique_ptr<DeviceChunk<float3> >(new DeviceChunk<float3>(ccNelements));


	thrust::reduce_by_key(
		ccresult_copy_ptr, ccresult_copy_ptr + dims.s(), 
		statistics_ptr,		
		resultComponentIds.get()->getTPtr(),
		resultComponentContributions.get()->getTPtr(),
		thrust::equal_to<int>(),
		thrust::plus<float3>());
	
	CompStatsF3 ret;
	ret.first = std::move(resultComponentIds);
	ret.second = std::move(resultComponentContributions);
	return ret;
}

void printStats(CurveProps& curParams, int nComps, ostream& stream){
	HostFloatChk hSurf(nComps), hVol(nComps), hSm(nComps);
	HostChunk<float3> hCog(nComps);
	curParams.surfContribs.second->copyHostN(hSurf, nComps);
	curParams.volContribs.second->copyHostN(hVol, nComps);
	curParams.cogContribs.second->copyHostN(hCog, nComps);
	curParams.smContribs.second->copyHostN(hSm, nComps);
	BOOST_LOG_TRIVIAL(info) << "Components: (n=" << nComps-1 << ")";
	for(int i = 1; i < nComps; i++){
		float cSurf = hSurf[i];
		float cVol = hVol[i];
		float3 cCog = hCog[i];
		float pPlasma = pow(cSurf, 3.0f/2.0f)/cVol;
		float cSm = hSm[i];
		BOOST_LOG_TRIVIAL(info) <<
				"\t(" << i <<
				"): volume=" << cVol <<
				", surface=" << cSurf <<
				", plasma=" << pPlasma <<
				", center of gravity=(" << cCog.x << "," << cCog.y << "," << cCog.z <<
				") second moment=" << cSm;
	}
	BOOST_LOG_TRIVIAL(info) << "Total: " << "volume=" << curParams.vol << ", surface=" << curParams.surf;
}

CurveProps computeStats(unique_ptr<DevFloatChk>& ls1, unique_ptr<DevIntChk>& ccResult, int nComps, GridParams gParams){
	DevFloatChk aux(gParams.gridSize.vol());

	CurveProps curParams;
	// Surface
	curParams.surf = getSurface_global(*ls1, aux, gParams.gridRes, gParams.gridSize);
	curParams.surfContribs = extractComponentStatistics(aux.getPtr(), *ccResult, nComps, gParams.gridSize.getd());

	//float* surfC = curParams.surfContribs.second->getPtr();
	//fill(device, surfC, surfC+curParams.surfContribs.second->getElements(), curParams.surf);

	// Volume
	curParams.vol = getVolume_global(*ls1, aux, gParams.gridRes, gParams.gridSize);
	curParams.volContribs = extractComponentStatistics(aux.getPtr(), *ccResult, nComps, gParams.gridSize.getd());

	//float* volC = curParams.volContribs.second->getPtr();
	//fill(device, volC, volC+curParams.volContribs.second->getElements(), curParams.vol);

	if(COMPUTE_ADVANCED_STATS){
		// CoG
		DeviceChunk<float3> auxf3(gParams.gridSize.vol());
		curParams.cog = getCOG(*ls1, auxf3, gParams.gridRes, gParams.gridSize);
		curParams.cogContribs = extractComponentStatistics(auxf3.getPtr(), *ccResult, nComps, gParams.gridSize.getd());

		// Second moment
		curParams.sm = getSecondMoment_global(*ls1, aux, *(curParams.cogContribs.second), *ccResult, gParams.gridRes, gParams.gridSize);
		curParams.smContribs = extractComponentStatistics(aux.getPtr(), *ccResult, nComps, gParams.gridSize.getd());
	}
	return curParams;
}
