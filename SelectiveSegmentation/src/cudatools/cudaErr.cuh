#ifndef CUDA_ERR_CUH
#define CUDA_ERR_CUH

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
extern void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true);

#endif
