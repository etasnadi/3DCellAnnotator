#include <stdio.h>
#include "cuda.h"

#include "cuda_runtime.h"     



////////////////////////////////////////////////////////////////////////////////
// Cuda error checking
////////////////////////////////////////////////////////////////////////////////

void SAFE_CALL(cudaError_t err){
	if(err != cudaSuccess){
		printf("Error: %s \n", cudaGetErrorString(err));
	}
}

void KERNEL_ERROR_CHECK(){
	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if(errSync != cudaSuccess){
		printf("\tSync kernel error: %s \n", cudaGetErrorString(errSync));
	}
	if(errAsync != cudaSuccess){
		printf("\tAsync kernel error: %s \n", cudaGetErrorString(errAsync));
	}
}

void KERNEL_ERROR_CHECK(char const *message){
	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if(errSync != cudaSuccess){
		printf("%s\n", message);
		printf("\tSync kernel error: %s \n", cudaGetErrorString(errSync));
	}
	if(errAsync != cudaSuccess){
		printf("%s\n", message);
		printf("\tAsync kernel error: %s \n", cudaGetErrorString(errAsync));
	}
}
