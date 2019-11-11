#ifndef EVOLVE_CUH
#define EVOLVE_CUH 1

#include "surfaceAndVolume.cuh"
#include "common.cuh"

void launchEvolve(dchunk_float& d_out, dchunk_float& d_in, dchunk_float& d_image, float* d_dataTerm, FltDer3D* imgDers, int3 gridToImageTranslation,
				dchunk<float3>& normals, dchunk<float>& K,
				int *d_nodeID,
				CurveProps& curParams,
                GridParams gridProps, Size3D imageDims,
                AlgParams params,
                int iter,
                Obj pref	/* preferred shape */);

__global__ void computeL(float* result, float* func_in, Size3D dims, float dh);
void computeDataTerm(float* dataTerm, float* image, Size3D imageSize);
void computeImgDerivatives(FltDer3D *der, float* image, Size3D imageSize);

std::pair<dchunk<Triangle>::uptr_t, dchunk_uint32::uptr_t> getFeasibleSet(dchunk_float& d_levelSet, dchunk_float& d_K, float Kthreshold,
		GridParams gParams, AlgParams aParams);

#endif
