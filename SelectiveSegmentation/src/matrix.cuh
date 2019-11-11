#ifndef MATRIX_CUH
#define MATRIX_CUH

#include "cudatools/function.cuh"

#define MATSIZE 3*3

template<typename T>
class Mat {
public:
	int w;
	int h;
	T d[MATSIZE];

	__device__ __host__ static Mat<T> cvec3(float3 f);
	__device__ __host__ static Mat<T> rvec3(float3 f);

	__device__ __host__ Mat(FltDer3D der);

	// Init matrix with zeros
	__device__ __host__ Mat(int aH, int aW);

	// Init with spec vals
	__device__ __host__ void initv(T val,int aH, int aW);
	__device__ __host__ void s(int i, int j, T val);
	__device__ __host__ T e(int i, int j);
	__device__ __host__ float3 getCvec3();
	__device__ __host__ float3 getRvec3();
	__device__ __host__ void pr();
	__device__ __host__ Mat<T> trans();
	__device__ __host__ operator float();
	__device__ __host__ T tr();
};

__device__ __host__ Mat<float> eye(int wh);
__device__ __host__ Mat<float> eye(int h, int w);
__device__ __host__ Mat<float> operator+(Mat<float> a, Mat<float> b);
__device__ __host__ Mat<float> operator-(Mat<float> a, Mat<float> b);
__device__ __host__ Mat<float> operator*(Mat<float> a, Mat<float> b);
__device__ __host__ Mat<float> operator*(Mat<float> a, float f);
__device__ __host__ Mat<float> operator*(float f, Mat<float> a);
__device__ __host__ Mat<float> operator*(float3 f, Mat<float> a);
__device__ __host__ Mat<float> operator*(Mat<float> a, float3 f);
__device__ __host__ float MdotR(Mat<float> mat1, Mat<float> mat2);

typedef Mat<float> FMat;

#endif
