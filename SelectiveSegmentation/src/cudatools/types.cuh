#ifndef TYPES_CUH
#define TYPES_CUH

#include <iostream>
//#include "cuda_runtime.h"

using namespace std;

#define EU Point3D(1, 0, 0)
#define EV Point3D(0, 1, 0)
#define EW Point3D(0, 0, 1)
#define ONES3 Point3D(1, 1, 1)
#define ZEROS3 Point3D(0, 0, 0)

#define IONES3 ( make_int3(1, 1, 1) )
#define IZEROS3 ( make_int3(0, 0, 0) )

#define FONES3 ( make_float3(1.0f, 1.0f, 1.0f) )
#define FZEROS3 ( make_float3(0.0f, 0.0f, 0.0f) )


#define _EPS 1E-15

typedef int2 Point2D;

typedef struct {
	int width;
	int height;
} Size2D;


class Point3D;
class Size3D;

class Point3D {
public:
	int x = 0;
	int y = 0;
	int z = 0;
	__device__ __host__ Point3D(const Point3D& p);
	__device__ __host__ Point3D();
	__device__ __host__ Point3D(int pX, int pY, int pZ);
	__device__ __host__ Point3D(int3 pP);
	__device__ __host__ Point3D(float3 pP);
	__device__ __host__ Point3D(Size3D pP);

	__device__ __host__ operator Size3D();
	__device__ __host__ operator float3();

	__host__ __device__ Point3D operator+ (const Point3D& o);
	__host__ __device__ Point3D operator- (const Point3D& o);
	__host__ __device__ int operator* (const Point3D& o);

	__host__ __device__ Point3D operator+ (const int o);
	__host__ __device__ Point3D operator- (const int o);
	__host__ __device__ Point3D operator*(const int m);

	__host__ __device__ Point3D cross(const Point3D o);

	__host__ __device__ bool operator== (const Point3D& b);
	__host__ __device__ bool operator!= (const Point3D& b);
	__host__ __device__ bool operator<= (const Point3D& b);
	__host__ __device__ bool operator>= (const Point3D& b);
	__host__ __device__ bool operator< (const Point3D& b);
	__host__ __device__ bool operator> (const Point3D& b);

	__host__ __device__ bool operator== (const int b);
	__host__ __device__ bool operator<= (const int b);
	__host__ __device__ bool operator>= (const int b);
	__host__ __device__ bool operator< (const int b);
	__host__ __device__ bool operator> (const int b);

	__host__ __device__ int3 getInt3();
	__host__ __device__ float3 getFloat3();

	__host__ __device__ Point3D abs();

	__host__ __device__ int sum();
	__host__ __device__ int min();
	__host__ __device__ int max();
};

ostream& operator<<(ostream& os, Point3D& p);

__host__ __device__ Size3D operator/(Size3D a, int b);
__host__ __device__ Size3D operator*(Size3D a, int b);

ostream& operator<<(ostream& os, Size3D& dt);

namespace commonns{
	class Dimensions {
	public:
		int dimX, dimY, dimZ;
		Dimensions();
		Dimensions(int a_dimX, int a_dimY, int a_dimZ);
		int s();
	};
}

class Size3D {
public:
	int width;
	int height;
	int depth;

	__host__ __device__ Size3D();
	__host__ __device__ Size3D(int a_w, int a_h, int a_d);
	__host__ __device__ Size3D(int3 pSize);
	__host__ __device__ int vol() const;
	__host__ __device__ int3 geti3();
	commonns::Dimensions getd();
};

class GpuConf {
protected:
	dim3 blockSize;
	dim3 gridSize;
public:
	dim3 block();
	dim3 grid();
};

class GpuConf3D : public GpuConf {
public:
	GpuConf3D(Size3D size, int a_blockSize);
};

class GpuConf2D : public GpuConf {
public:
	GpuConf2D(Size2D size, int a_blockSize);
};

class GpuConf1D : public GpuConf {
public:
	GpuConf1D(int size, int blockSize);
};

class WorkManager {
private:
	GpuConf _conf;

public:
	WorkManager(Size3D size, int blockSize){
		_conf = GpuConf3D(size, blockSize);
	}

	WorkManager(Size2D size, int blockSize){
		_conf = GpuConf2D(size, blockSize);
	}

	WorkManager(int size, int blockSize){
		_conf = GpuConf1D(size, blockSize);
	}

	GpuConf conf() const {
		return _conf;
	}

};

class ThreadInfo {
public:
	int3 getThreadId(){
		int x = blockIdx.x*blockDim.x + threadIdx.x;
		int y = blockIdx.y*blockDim.y + threadIdx.y;
		int z = blockIdx.z*blockDim.z + threadIdx.z;
		return make_int3(x, y, z);
	}
};

/*
__host__ __device__ Mat3(float3 c1, float3 c2, float3 c3){

}
*/

/*
__host__ __device__ float3 operator*(float3 f, Derivative3D<float> m){
	return make_float3(
			f.x*m.xx + f.y*m.xy + f.z*m.xz,
			f.x*m.xy + f.y*m.yy + f.z*m.yz,
			f.x*m.xz + f.y*m.yz + f.z*m.zz
			);
}

__host__ __device__ float3 operator*(Derivative3D<float> m, float3 f){
	return make_float3(
			f.x*m.xx + f.y*m.xy + f.z*m.xz,
			f.x*m.xy + f.y*m.yy + f.z*m.yz,
			f.x*m.xz + f.y*m.yz + f.z*m.zz
			);
}
*/


__host__ __device__ float _mul(float3 a, float3 b);

extern __device__ int getLocalThreadIdx3D();
__device__ int getLocalElem3D(int idx);
extern __device__ Point3D getThread3D();
extern __device__ Point2D getThread2D();
extern __device__ int getThread();
__host__ __device__ Point3D getlinTrans(Size3D size);
extern int divUp(int a, int b);

// ===================== float3 overloads

__host__ __device__ float3 make_float3(const int3 a);
__host__ __device__ float3 make_float3(const float f);

__host__ __device__ float3 operator+(const float3 a, const float3 b);
__host__ __device__ float3 operator-(const float3 a, const float3 b);
__host__ __device__ float operator*(const float3 a, const float3 b);
__host__ __device__ float3& operator+=(float3& a, const float3 b);
__host__ __device__ float3& operator-=(float3& a, const float3 b);

__host__ __device__ float3 operator*(const float3 a, const float b);
__host__ __device__ float3 operator/(const float3 a, const float b);

__host__ __device__ float3 operator*(const float a, const float3 b);

__host__ __device__ bool operator<(float3 a, float3 b);
__host__ __device__ bool operator>(float3 a, float3 b);


// Hadamard-product
__host__ __device__ float3 hp(const float3 a, const float3 b);
__host__ __device__ float3 cross(const float3 u, const float3 b);
__host__ __device__ float3 homDiv(const float4 p);

__host__ __device__ float sum(const float3 p);
__host__ __device__ float mul(const float3 v);

__host__ __device__ float3 fabs(const float3 v);
__host__ __device__ float L2(const float3 p);
__host__ __device__ int3 round(const float3 f);

ostream& operator<<(ostream& os, float3& f3);

// ===================== int3 overloads

// Construction
__host__ __device__ int3 make_int3(const int a);
__host__ __device__ int3 make_int3(const float3 f);
__device__ int3 f3toi3_rd(float3 f);

// Arithmetic

__host__ __device__ int3 operator+(const int3 a, const int3 b);
__host__ __device__ int3 operator-(const int3 a, const int3 b);
__host__ __device__ int operator*(const int3 a, const int3 b);
__host__ __device__ int3& operator+=(int3& a, const int3 b);
__host__ __device__ int3& operator-=(int3& a, const int3 b);
__host__ __device__ int3& operator*=(int3& a, const int3 b);

__host__ __device__ int3 operator+(const int3 a, const int b);
__host__ __device__ int3 operator-(const int3 a, const int b);
__host__ __device__ int3 operator*(const int3 a, const int b);
__host__ __device__ int3& operator+=(int3& a, const int b);
__host__ __device__ int3& operator-=(int3& a, const int b);
__host__ __device__ int3& operator*=(int3& a, const int b);

__host__ __device__ int3 operator+(const int a, const int3 b);
__host__ __device__ int3 operator-(const int a, const int3 b);
__host__ __device__ int3 operator*(const int a, const int3 b);
__host__ __device__ int3& operator+=(const int a, int3& b);
__host__ __device__ int3& operator-=(const int a, int3& b);
__host__ __device__ int3& operator*=(const int a, int3& b);

__host__ __device__ int mul(int3 v);
ostream& operator<<(ostream& os, int3& i);

// Comparison
__host__ __device__ bool operator<(const int3 a, const int3 b);
__host__ __device__ bool operator<=(const int3 a, const int3 b);
__host__ __device__ bool operator>(const int3 a, const int3 b);
__host__ __device__ bool operator>=(const int3 a, const int3 b);

// Other

__host__ __device__ int3 getRowMajorTransform(int dimX, int dimY);
__host__ __device__ int3 getRowMajorTransform(Size3D size);

typedef int2 IntPair;

// float4 overloads

__host__ __device__ float4 operator*(float4 a, float b);
__host__ __device__ float4 operator*(float b, float4 a);
__host__ __device__ float4 operator+(float4 a, float4 b);

#endif
