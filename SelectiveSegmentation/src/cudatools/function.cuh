#ifndef FUNCTION_CUH
#define FUNCTION_CUH 1

#include "types.cuh"

template<typename Size, typename Point>
class AddrLookup;

// Address lookup 3D, row major format
template<>
class AddrLookup<Size3D, Point3D> {
private:
	Size3D s;
public:
	__device__ __host__ AddrLookup(Size3D pS) : s(pS) {}
	__device__ __host__ int look(Point3D point) const;
};

template<typename Size, typename Point>
class AddrLookupView;

template<>
class AddrLookupView<Size3D, Point3D> {
private:
	Size3D s;
	uint8_t dimSel;
public:
	__device__ __host__ AddrLookupView(Size3D pS, uint8_t a_dimSel) : s(pS), dimSel(a_dimSel) {}
	__device__ __host__ int look(Point3D point);
};

/*
template<>
class AddrLookupView<Size3D, Point3D> {
private:
	Size3D s;
	uint8_t dim;
public:
	__device__ __host__ AddrLookupView(Size3D pS, uint8_t a_dim) : s(pS), dim(a_dim) {}
	__device__ __host__ int look(Point3D point);
};
*/

// Address lookup 1D
template<>
class AddrLookup<int, int> {
private:
	int s;
public:
	__device__ __host__ AddrLookup(int pS) : s(pS) {}
	__device__ __host__ int look(int point) const {
		return point;
	}
};

template<>
class AddrLookup<Size2D, Point2D> {
private:
	Size2D s;
public:
	__device__ __host__ AddrLookup(Size2D pS) : s(pS) {}
	__device__ __host__ int look(Point2D point) const {
		return s.width*point.y + point.x;
	}
};

template<typename T, typename S, typename P, typename Lookup=AddrLookup<S, P>>
class FuncBase {
protected:
	T* func;
	S size;
	Lookup lookup;
public:
	typedef T data_type;
	typedef S size_type;
	typedef P index_type;

	__device__ __host__ FuncBase() : func(nullptr){

	}

	__device__ __host__ FuncBase(S pSize, T* pFunc) : size(pSize), func(pFunc), lookup(Lookup(pSize)){}
	__device__ __host__ T* getData(){
		return func;
	}

	__device__ __host__ S getSize(){
		return size;
	}

	__device__ __host__ Lookup getLookupFunction(){
		return lookup;
	}

	__device__ __host__ T& operator[] (const P p) const {
		return func[lookup.look(p)];
	}
};

template<typename T>
class Func3D : public FuncBase<T, Size3D, Point3D> {
public:
	__device__ __host__ Func3D() {}
	__device__ __host__ Func3D(int3 pSize, T* pFunc) : FuncBase<T, Size3D, Point3D>(pSize, pFunc) {}
	__device__ __host__ Func3D(Size3D pSize, T* pFunc) : FuncBase<T, Size3D, Point3D>(pSize, pFunc) {}
};

template<typename T>
class Derivative3D {
public:
	T x, y, z;
	T xx, yy, zz;
	T xy, xz, yz;

	__device__ __host__ Derivative3D() : x(0), y(0), z(0), xx(0), yy(0), zz(0), xy(0), xz(0), yz(0) {};
	__device__ __host__ float3 grad(){
		return make_float3(x, y, z);
	}
	// Norm of the gradient L2
	__device__ float calcGradLen() {
		return L2(grad()) + _EPS;
	}

	// Norm of the gradient L1
	__device__ float calcGradLenL1() {
		return sum(grad()) + _EPS;
	}
};

template<typename T>
__device__ __host__ Derivative3D<T> operator+(Derivative3D<T> a, Derivative3D<T> b){
	Derivative3D<T> ret;
	ret.x = a.x + b.x;
	ret.y = a.y + b.y;
	ret.z = a.z + b.z;

	ret.xx = a.xx + b.xx;
	ret.yy = a.yy + b.yy;
	ret.zz = a.zz + b.zz;;

	ret.xy = a.xy + b.xy;
	ret.xz = a.xz + b.xz;
	ret.yz = a.yz + b.yz;
	return ret;
}

__device__ __host__ float interpolate3D(const float3 pt, const Func3D<float>& levelSet);
__device__ __host__ float getVertexWeight(int3 cubeVertex, float3 localCubePt);

template<>
class Func3D<float> : public FuncBase<float, Size3D, Point3D> {
public:
	__host__ __device__ Func3D(int3 pSize, float* pFunc) : FuncBase<float, Size3D, Point3D>(Size3D(pSize), pFunc) {}
	__host__ __device__ Func3D(Size3D pSize, float* pFunc) : FuncBase<float, Size3D, Point3D>(pSize, pFunc) {}

	// First order central differences
	__device__ float dx(Point3D p, float dh){
		return (FuncBase::operator[](p+EU)-FuncBase::operator[](p-EU))/(2*dh);
	}

	__device__ float dy(Point3D p, float dh){
		return (FuncBase::operator[](p+EV)-FuncBase::operator[](p-EV))/(2*dh);
	}

	__device__ float dz(Point3D p, float dh){
		return (FuncBase::operator[](p+EW)-FuncBase::operator[](p-EW))/(2*dh);
	}

	// Second order differences: backward difference of the first order forward
	__device__ float dxx(Point3D p, float dh){
		return (FuncBase::operator[](p+EU) - 2*FuncBase::operator[](p) + FuncBase::operator[](p-EU))/pow(dh, 2);
	}

	__device__ float dyy(Point3D p, float dh){
		return (FuncBase::operator[](p+EV) - 2*FuncBase::operator[](p) + FuncBase::operator[](p-EV))/pow(dh, 2);
	}

	__device__ float dzz(Point3D p, float dh){
		return (FuncBase::operator[](p+EW) - 2*FuncBase::operator[](p) + FuncBase::operator[](p-EW))/pow(dh, 2);
	}

	// Second order mixed differences: central-central
	__device__ float dxy(Point3D p, float dh) {
		return (FuncBase::operator[](p+EU+EV) - FuncBase::operator[](p-EU+EV) - FuncBase::operator[](p+EU-EV)  + FuncBase::operator[](p-EU-EV))/(4*pow(dh, 2));
	}

	__device__ float dxz(Point3D p, float dh) {
		return (FuncBase::operator[](p+EU+EW) - FuncBase::operator[](p-EU+EW) - FuncBase::operator[](p+EU-EW)  + FuncBase::operator[](p-EU-EW))/(4*pow(dh, 2));
	}

	__device__ float dyz(Point3D p, float dh) {
		return (FuncBase::operator[](p+EV+EW) - FuncBase::operator[](p-EV+EW) - FuncBase::operator[](p+EV-EW)  + FuncBase::operator[](p-EV-EW))/(4*pow(dh, 2));
	}

	__device__ Derivative3D<float> der(Point3D p, float dh){
		Derivative3D<float> der;
		der.x = dx(p, dh);
		der.y = dy(p, dh);
		der.z = dz(p, dh);

		der.xx = dxx(p, dh);
		der.yy = dyy(p, dh);
		der.zz = dzz(p, dh);

		der.xy = dxy(p, dh);
		der.xz = dxz(p, dh);
		der.yz = dyz(p, dh);
		return der;
	}

	__device__ float3 grad(Point3D p, float dh){
		return make_float3(dx(p, dh), dy(p, dh), dz(p, dh));
	}

	__device__ float3 normal(Point3D p, float dh){
		float3 _grad = this->grad(p, dh);
		return _grad/L2(_grad);
	}

	__device__ __host__ float at(float3 pt) const {
		return interpolate3D(pt, *this);
	}

	__device__ __host__ float operator[] (const float3 pt) const {
		return interpolate3D(pt, *this);
	};

	__device__ __host__ float& operator[] (const Point3D p) const {
		return FuncBase::operator [](p);
	}
};

__device__ __host__ float3 interpolate3D(const float3 pt, const Func3D<float3>& func);

template<>
class Func3D<float3> : public FuncBase<float3, Size3D, Point3D> {
public:
	__host__ __device__ Func3D(int3 pSize, float3* pFunc) : FuncBase<float3, Size3D, Point3D>(Size3D(pSize), pFunc) {}
	__host__ __device__ Func3D(Size3D pSize, float3* pFunc) : FuncBase<float3, Size3D, Point3D>(pSize, pFunc) {}
	__device__ __host__ float3 at(float3 pt) const {
		return interpolate3D(pt, *this);
	}

	__device__ __host__ float3 operator[] (float3 pt) const {
		return interpolate3D(pt, *this);
	};

	__device__ __host__ float3& operator[] (const Point3D p) const {
		return FuncBase::operator [](p);
	}
};

template<typename T>
class Func1D : public FuncBase<T, int, int> {
public:
	__device__ __host__ Func1D(int pSize, T* pFunc) : FuncBase<T, int, int>(pSize, pFunc) {}
};

template<typename T>
class Func2D : public FuncBase<T, Size2D, Point2D> {
public:
	__device__ __host__ Func2D(Size2D pSize, T* pFunc) : FuncBase<T, Size2D, Point2D>(pSize, pFunc) {}
};


typedef Func3D<float> FltFunc3D;
typedef Derivative3D<float> FltDer3D;
typedef Func3D<int> IntFunc3D;

typedef Func2D<float> FltFunc2D;
typedef Func2D<int> IntFunc2D;

typedef Func1D<float> FltFunc1D;
typedef Func1D<int> IntFunc1D;

template<class T>
using func3 = Func3D<T>;

template<class T>
using func2 = Func2D<T>;

template<class T>
using func = Func1D<T>;

using func3_f = func3<float>;
using func2_f = func2<float>;
using func_f = func<float>;

using func3_f3 = func3<float>;
using func2_f3 = func2<float>;
using func_f3 = func<float>;

using func3_uint32 = func3<uint32_t>;
using func2_uint32 = func2<uint32_t>;
using func_uint32 = func<uint32_t>;



#endif
