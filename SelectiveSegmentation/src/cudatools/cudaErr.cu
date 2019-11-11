#include "cudaErr.cuh"

#include <iostream>




extern bool erro;

using namespace std;

void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess)
	{
		std::cout << "Cuda error: " << cudaGetErrorString(code) << " (" << file << ":" << line << ")" << std::endl;
		erro = true;
		//throw 20;
		//if (abort) exit(code);
	}
}
