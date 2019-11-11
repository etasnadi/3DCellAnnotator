#include <iostream>
#include "types.cuh"

using namespace std;

// Point3D impl

__device__ __host__ Point3D::Point3D(const Point3D& p) = default;
__device__ __host__ Point3D::Point3D() = default;
__device__ __host__ Point3D::Point3D(int pX, int pY, int pZ) : x(pX), y(pY), z(pZ){}
__device__ __host__ Point3D::Point3D(int3 pP) : x(pP.x), y(pP.y), z(pP.z){}
__device__ __host__ Point3D::Point3D(float3 pP) : x(int(pP.x)), y(int(pP.y)), z(int(pP.z)){}
__device__ __host__ Point3D::Point3D(Size3D pP) : x(pP.width), y(pP.height), z(pP.depth){}

__device__ __host__  Point3D::operator Size3D() {
	return Size3D(x, y, z);
}

__device__ __host__  Point3D::operator float3(){
	return make_float3(float(x), float(y), float(z));
}

// Point3D op int

__host__  __device__ Point3D Point3D::operator+ (const int o){
	return Point3D(x+o, y+o, z+o);
}

__host__  __device__ Point3D Point3D::operator- (const int o){
	return operator+(-o);
}

__host__ __device__ Point3D Point3D::operator*(const int m){
	return Point3D(x*m, y*m, z*m);
}

// Point3D op Point3D

__host__  __device__ Point3D Point3D::operator+ (const Point3D& o){
	return Point3D(x+o.x, y+o.y, z+o.z);
}

__host__ __device__ Point3D Point3D::operator- (const Point3D& o){
	return Point3D(x-o.x, y-o.y, z-o.z);
}

__host__ __device__ int Point3D::operator*(const Point3D& o){
	return x*o.x + y*o.y + z*o.z;
}

// Point3D op Point3D

__host__ __device__ bool Point3D::operator== (const Point3D& b){
	return x == b.x && y == b.y && z == b.z;
}

__host__ __device__ bool Point3D::operator!= (const Point3D& b){
	return x != b.x && y != b.y && z != b.z;
}

__host__ __device__ bool Point3D::operator<= (const Point3D& b){
	return x <= b.x && y <= b.y && z <= b.z;
}

__host__ __device__ bool Point3D::operator>= (const Point3D& b){
	return x >= b.x && y >= b.y && z >= b.z;
}

__host__ __device__ bool Point3D::operator< (const Point3D& b){
	return x < b.x && y < b.y && z < b.z;
}

__host__ __device__ bool Point3D::operator> (const Point3D& b){
	return x > b.x && y > b.y && z > b.z;
}

ostream& operator<<(ostream& os, Point3D& p){
	os << "(" << p.x << "," << p.y << "," << p.z << ")";
	return os;
}

// Point3D op int

__host__ __device__ bool Point3D::operator== (int b){
		return x == b && y == b && z == b;
	}

__host__ __device__ bool Point3D::operator<= (const int b){
	return operator<=(ONES3*b);
}

__host__ __device__ bool Point3D::operator>= (const int b){
	return operator>=(ONES3*b);
}

__host__ __device__ bool Point3D::operator< (const int b){
	return operator<(ONES3*b);
}

__host__ __device__ bool Point3D::operator> (const int b){
	return operator>(ONES3*b);
}

__host__ __device__ int3 Point3D::getInt3(){
	return make_int3(x, y, z);
}

__host__ __device__ float3 Point3D::getFloat3(){
	return make_float3(float(x), float(y), float(z));
}

__host__ __device__ Point3D Point3D::abs(){
	return Point3D(::abs(x), ::abs(y), ::abs(z));
}

__host__ __device__ int Point3D::sum(){
	return x + y + z;
}
__host__ __device__ int Point3D::min(){
	return ::min(::min(x, y), z);
}
__host__ __device__ int Point3D::max(){
	return ::max(::max(x, y), z);
}

// Size 3D impl

ostream& operator<<(ostream& os, Size3D& o){
	os << "(" << o.width << "," << o.height << "," << o.depth << ")";
	return os;
}

//#define Point3D() 3Dto1Dtrans

__host__ __device__ Size3D::Size3D() : width(0), height(0), depth(0){};

__host__ __device__ Size3D::Size3D(int a_w, int a_h, int a_d) : width(a_w), height(a_h), depth(a_d){};

__host__ __device__ Size3D::Size3D(int3 pSize) : width(pSize.x), height(pSize.y), depth(pSize.z){};

__host__ __device__ Size3D operator/(Size3D s, const int o){
	return Size3D(s.width/o, s.height/o, s.depth/o);
}

__host__ __device__ Size3D operator*(Size3D s, const int o){
	return Size3D(s.width*o, s.height*o, s.depth*o);
}

__host__ __device__ int Size3D::vol() const {
	return width*height*depth;
}

__host__ __device__ int3 Size3D::geti3(){
	return make_int3(width, height, depth);
}

// Dimensions...

__host__ commonns::Dimensions Size3D::getd(){
	return commonns::Dimensions(width, height, depth);
}

commonns::Dimensions::Dimensions(int a_dimX, int a_dimY, int a_dimZ) : dimX(a_dimX), dimY(a_dimY), dimZ(a_dimZ){
}


int commonns::Dimensions::s(){
	return dimX*dimY*dimZ;
}

commonns::Dimensions::Dimensions(){
}

// Gpu conf...

int divUp(int a, int b){
	return (a + b - 1) / b;
}

GpuConf3D::GpuConf3D(Size3D size, int a_blockSize){
	blockSize = dim3(a_blockSize, a_blockSize, a_blockSize);
	gridSize = dim3(
		divUp(size.width, a_blockSize),
		divUp(size.height, a_blockSize),
		divUp(size.depth, a_blockSize)
	);
}

GpuConf2D::GpuConf2D(Size2D size, int a_blockSize){
	blockSize = dim3(a_blockSize, a_blockSize, a_blockSize);
	gridSize = dim3(
		divUp(size.width, a_blockSize),
		divUp(size.height, a_blockSize)
	);
}

GpuConf1D::GpuConf1D(int size, int a_blockSize){
	blockSize = dim3(a_blockSize);
	gridSize = dim3(divUp(size, a_blockSize));
}

dim3 GpuConf::block() {
	return blockSize;
}

dim3 GpuConf::grid() {
	return gridSize;
}

__host__ __device__ Point3D getlinTrans(Size3D size){
	return Point3D(1, size.width, size.width*size.height);
}

extern __device__ int getLocalThreadIdx3D(){
	return threadIdx.x + threadIdx.y*blockDim.y + threadIdx.z*blockDim.z;
}

__device__ int getLocalElem3D(int index){
	int tid = getLocalThreadIdx3D();
	int tbSize = blockDim.x*blockDim.y*blockDim.z;
	return index * tbSize + tid;
}

extern __device__  Point3D getThread3D(){
	Point3D p;
	p.x = blockIdx.x*blockDim.x + threadIdx.x;
	p.y = blockIdx.y*blockDim.y + threadIdx.y;
	p.z = blockIdx.z*blockDim.z + threadIdx.z;
	return p;
}

extern __device__ Point2D getThread2D(){
	Point2D p;
	p.x = blockIdx.x*blockDim.x + threadIdx.x;
	p.y = blockIdx.y*blockDim.y + threadIdx.y;
	return p;
}

extern __device__ int getThread(){
	return blockIdx.x*blockDim.x + threadIdx.x;
}

// ===================== float3 overloads

__host__ __device__ float3 make_float3(const int3 a){
    return make_float3(float(a.x), float(a.y), float(a.z));
}

__host__ __device__ float3 make_float3(const float f){
	return make_float3(f, f, f);
}

__host__ __device__ float3 operator+(const float3 a, const float3 b){
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ float3 operator-(const float3 a, const float3 b){
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float operator*(const float3 a, const float3 b){
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float3 operator/(const float3 a, const float3 b){
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__ __device__ float3& operator+=(float3& a, const float3 b){
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

__host__ __device__ float3& operator-=(float3& a, const float3 b){
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

__host__ __device__ float3 operator*(const float3 a, const float b){
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ float3 operator/(const float3 a, const float b){
	return make_float3(a.x/b, a.y/b, a.z/b);
}

__host__ __device__ float3 operator*(const float a, const float3 b){
	return make_float3(a*b.x, a*b.y, a*b.z);
}

__host__ __device__ bool operator<(float3 a, float3 b){
	return a.x < b.x && a.y < b.y && a.z < b.z;
}

__host__ __device__ bool operator>(float3 a, float3 b){
	return a.x > b.x && a.y > b.y && a.z > b.z;
}

// Misc ops

__host__ __device__ float3 hp(const float3 a, const float3 b){
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ float3 cross(const float3 u, const float3 v){
	return make_float3(u.y*v.z-u.z*v.y, u.z*v.x-u.x*v.z, u.x*v.y-u.y*v.x);
}

__host__ __device__ float3 homDiv(const float4 p){
    return make_float3(p.x/p.w, p.y/p.w, p.z/p.w);
}

__host__ __device__ float sum(const float3 p){
    return p.x+p.y+p.z;
}

__host__ __device__ float mul(const float3 v){
	return v.x*v.y*v.z;

}

__host__ __device__ float L2(const float3 p){
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

__host__ __device__ float3 fabs(const float3 v){
   return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}

__host__ __device__ int3 round(const float3 f){
	return make_int3(::round(f.x), ::round(f.y), ::round(f.z));
}


ostream& operator<<(ostream& os, float3& f3){
	os << "(" << f3.x << "," << f3.y << "," << f3.z << ")";
	return os;
}

// ===================== int3 overloads

// Contruction

__host__ __device__ int3 make_int3(const float3 f){
	return make_int3(int(f.x), int(f.y), int(f.z));
}

__host__ __device__ int3 make_int3(const int a){
	return make_int3(a, a, a);
}

__device__ int3 f3toi3_rd(float3 f){
	return make_int3(__float2int_rd(f.x), __float2int_rd(f.y), __float2int_rd(f.z));
}

// Arithmetic

// int3 op int3 +,-,*,+=,-=,*=

__host__ __device__ int3 operator+(const int3 a, const int3 b){
	return make_int3(
			a.x + b.x,
			a.y + b.y,
			a.z + b.z
	);
}

__host__ __device__ int3 operator-(const int3 a, const int3 b){
	return make_int3(
			a.x-b.x,
			a.y-b.y,
			a.z-b.z
	);
}

__host__ __device__ int operator*(const int3 a, const int3 b){
	return
			a.x*b.x +
			a.y*b.y +
			a.z*b.z;
}

__host__ __device__ int3& operator+=(int3& a, const int3 b){
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

__host__ __device__ int3& operator-=(int3& a, const int3 b){
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

__host__ __device__ int3& operator*=(int3& a, const int3 b){
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}

// int3 op int +,-,*,+=,-=,*=

__host__ __device__ int3 operator+(const int3 a, const int b){
	return make_int3(
			a.x + b,
			a.y + b,
			a.z + b
	);
}

__host__ __device__ int3 operator-(const int3 a, const int b){
	return make_int3(
			a.x-b,
			a.y-b,
			a.z-b
	);
}

__host__ __device__ int3 operator*(const int3 a, const int b){
	return make_int3(
			a.x*b,
			a.y*b,
			a.z*b
	);
}

__host__ __device__ int3& operator+=(int3& a, const int b){
	a.x += b;
	a.y += b;
	a.z += b;
	return a;
}

__host__ __device__ int3& operator-=(int3& a, const int b){
	a.x -= b;
	a.y -= b;
	a.z -= b;
	return a;
}

__host__ __device__ int3& operator*=(int3& a, const int b){
	a.x *= b;
	a.y *= b;
	a.z *= b;
	return a;
}

// int op int3 +,-,*

__host__ __device__ int3 operator+(const int a, const int3 b){
	return b+a;
}

__host__ __device__ int3 operator-(const int a, const int3 b){
	return make_int3(
			a-b.x,
			a-b.y,
			a-b.z
	);
}

__host__ __device__ int3 operator*(const int a, const int3 b){
	return b*a;
}

__host__ __device__ int3& operator+=(const int a, int3& b){
	b.x += a;
	b.y += a;
	b.z += a;
	return b;
}

__host__ __device__ int3& operator-=(const int a, int3& b){
	b.x -= a;
	b.y -= a;
	b.z -= a;
	return b;
}

__host__ __device__ int3& operator*=(const int a, int3& b){
	b.x *= a;
	b.y *= a;
	b.z *= a;
	return b;
}

__host__ __device__ int mul(int3 v){
	return v.x * v.y * v.z;
}

ostream& operator<<(ostream& os, int3& i){
	os << "(" << i.x << "," << i.y << "," << i.z << ")";
	return os;
}

// Comparison

// int3 op int3 >,<,>=,<=

__host__ __device__ bool operator<(const int3 a, const int3 b){
	return (a.x < b.x && a.y < b.y && a.z < b.z);
}

__host__ __device__ bool operator>(const int3 a, const int3 b){
	return (a.x > b.x && a.y > b.y && a.z > b.z);
}

__host__ __device__ bool operator<=(const int3 a, const int3 b){
	return (a.x <= b.x && a.y <= b.y && a.z <= b.z);
}

__host__ __device__ bool operator>=(const int3 a, const int3 b){
	return (a.x >= b.x && a.y >= b.y && a.z >= b.z);
}

// Other

__host__ __device__ int3 getRowMajorTransform(int dimX, int dimY){
	return make_int3(1, dimX, dimX*dimY);
}

__host__ __device__ int3 getRowMajorTransform(Size3D size){
	return make_int3(1, size.width, size.width*size.height);
}

// ===================== float4 overloads
__host__ __device__ float4 operator*(float4 a, float b){
    return make_float4(
    		a.x*b,
    		a.y*b,
    		a.z*b,
    		a.w*b);
}

__host__ __device__ float4 operator*(float b, float4 a){
    return make_float4(
    		a.x*b,
    		a.y*b,
    		a.z*b,
    		a.w*b);
}

__host__ __device__ float4 operator+(float4 a, float4 b){
    return make_float4(
    		a.x+b.x,
    		a.y+b.y,
    		a.z+b.z,
    		a.w+b.w);
}
