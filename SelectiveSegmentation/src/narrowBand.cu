#include "evolve.cuh"
#include "common.cuh"
#include "cudatools/errorHandling.cuh"

////////////////////////////////////////////////////////////////////////////////
// Narrow band functions
////////////////////////////////////////////////////////////////////////////////

// Poor man's narrow band

/**
 *
 * Roughly approximates the distance of the isosurface.
 * i0, j0, k0: the reference point
 * bandSize should be positive or 0.
 */
__device__ int calculateIsoSurfaceDistance(float* d_in, int bandSize, int i0, int j0, int k0, int dimX, int dimY, int dimZ){

	// min_dist is the distance of the first point that has a level-set value with different sign
	int min_dist = bandSize*2;
	//int mi, mj, mk;

	float levelSetValueRef = d_in[i0 + j0*dimX + k0*dimX*dimY];
	int pipetteSize = bandSize+1;
	for(int i = -pipetteSize; i <= pipetteSize; i++){
		for(int j = -pipetteSize; j <= pipetteSize; j++){
			for(int k = -pipetteSize; k <= pipetteSize; k++){
				// The other point
				int i_other = i0+i;
				int j_other = j0+j;
				int k_other = k0+k;

				// The point is not the same as the reference point and the other point is in the picture,
				if(i != 0 && j != 0 && k!= 0 && isInnerPoint(i_other, j_other, k_other, dimX, dimY, dimZ)){
					int dist = getDist(i, j, k);
					if(dist < min_dist){
						float levelSetValueOther = d_in[i_other + j_other*dimX + k_other*dimX*dimY];
						if(sgn(levelSetValueOther) != sgn(levelSetValueRef)){
							min_dist = dist;
						}
					}
				}
			}
		}
	}
	min_dist--;
	if(min_dist > bandSize){
		return -1;
	}else{
		return min_dist;
	}

}

__global__ void reinitLevelSetNarrowBand(float* d_out, float* d_in, int bandSize, int dimX, int dimY, int dimZ, float dh, float dt){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	if(isInnerPoint(i, j, k, dimX, dimY, dimZ)){
		int d = calculateIsoSurfaceDistance(d_in, bandSize, i, j, k, dimX, dimY, dimZ);
		if(d < bandSize && d != -1){
			//d_out[idx] = -225.0;
			//reinitPoint(d_out, d_in, i, j, k, dimX, dimY, dimZ, dh, dt);
		}
	}

}

__global__ void initialiseNarrowBand(float* d_out, float* d_in, int bandSize, float boundaryValue, int dimX, int dimY, int dimZ){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	int idx = i + j*dimX + k*dimX*dimY;

	if(isInnerPoint(i, j, k, dimX, dimY, dimZ)){
		int d = calculateIsoSurfaceDistance(d_in, bandSize, i, j, k, dimX, dimY, dimZ);
		d_out[idx] = 0;

		if(d == bandSize){ // The point is the part of the outer/inner narrow band
			if(d_in[idx] < 0){
				d_out[idx] = -boundaryValue;
			}else{
				d_out[idx] = boundaryValue;
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// Nearest object
////////////////////////////////////////////////////////////////////////////////

__device__ int nearestComponentToVoxel(int *d_nodeID, int bandSize, int i0, int j0, int k0, int dimX, int dimY, int dimZ)
{
	int minDist = 99999999;
	int minID = -1;

	for(int i = -bandSize; i <= bandSize; i++){
		for(int j = -bandSize; j <= bandSize; j++){
			for(int k = -bandSize; k <= bandSize; k++){
				if(isValidPoint(i0+i, j0+j, k0+k, dimX, dimY, dimZ)) {
					int tempID = d_nodeID[i0+i + (j0+j)*dimX + (k0+k)*dimX*dimY];
					if(tempID != -1 && i*i+j*j+k*k <= minDist){
						minID = tempID;
						minDist = i*i+j*j+k*k;
					}
				}
			}
		}	
	}
	return minID;
}

__global__ void nearestComponentKernel(int *d_out, int *d_nodeID, int bandSize, int dimX, int dimY, int dimZ)
{
	// calculating 3D indices
	// (i, j, k) is the 3D index of the data array
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;
	
	if(i >= 0 && j >= 0 && k >= 0 && i < dimX && j < dimY && k < dimZ){
		d_out[i+j*dimX+k*dimX*dimY] = nearestComponentToVoxel(d_nodeID, bandSize, i, j, k, dimX, dimY, dimZ);
	}
}

void findClosestObjectNB(int *d_out, int *d_nodeID, int bandSize, Size3D gridDims)
{
	int dimX = gridDims.width;
	int dimY = gridDims.height;
	int dimZ = gridDims.depth;

	dim3 blockSize(3,3,3);
	dim3 gridSize(divUp(dimX,3), divUp(dimY,3), divUp(dimZ,3));
	nearestComponentKernel<<<gridSize, blockSize>>>(d_out, d_nodeID, bandSize, dimX, dimY, dimZ);
	KERNEL_ERROR_CHECK("nearestComponentKernel error");
}
