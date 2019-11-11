/*
 * localRegion.cuh
 *
 *  Created on: 2018 nov. 2
 *      Author: ervin
 */

#ifndef LOCALREGION_CUH_
#define LOCALREGION_CUH_

#include "common.cuh"
#include "matrix.cuh"

#define OUT_PT_INT 255

class LocalRegion {
private:
	int3 extent;
public:
	LocalRegion(int w, int h, int d);
	__host__ __device__ int3 getExtent();
	__host__ __device__ int3 getSize();
};

typedef struct {
	float3 n;
	float3 e1;
	float3 e2;
} lr_basis;


__device__ lr_basis getLrBasis(float3 grad);
__device__ Point3D getImageCoord(int3 PointLoc, lr_basis basis, Point3D S, int scal);
__host__ __device__ int3 getLrSize(int3 lrExtent);

extern __device__ FMat calcNormNab(FltFunc3D& f_in, Point3D p);

extern __device__ float calcLR_(
		float* d_image,
		int3 imgDims,
		FltDer3D* d_ders,
		GridParams gProps,
		Point3D S,
		float* d_level,
		float3 grad,
		FMat NNab,
		AlgParams aParams);

#endif /* LOCALREGION_CUH_ */
