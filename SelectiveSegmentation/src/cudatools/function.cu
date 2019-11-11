#ifndef FUNCTION_CU
#define FUNCTION_CU 1

#include "function.cuh"
#include "cuda.h"
#include "cuda_runtime.h"    


__device__ __host__ int AddrLookup<Size3D, Point3D>::look(Point3D point) const {
	return (point.x +
	(point.y * s.width) +
	(point.z * s.width * s.height));
}

/*
 *	Returns the weight of a cube vertex for the 3D interpolation.
 *	The cube has 8 points, they can be addressed with an {0,1}^3 vector.
 *	1D interpolation: x0=0, x1=1; we have a value phi(x0) and phi(x1).
 *	We want to compute the first approximation of phi(x): ~phi(x).
 *	(Classic linear interpolation). Find the line l(_x)=a*_x+b that fullfils the
 *	Requirement: l(x0)=phi(x0) and l(x1)=phi(x1), so:
 *	b=phi(x0)
 *	a=dy/dx=[phi(x1)-phi(x0)]/(x1-x0)
 *	So: l(x)={[phi(x1)-phi(x0)]/(x1-x0)}*x+phi(x0)
 *	Since x1=1 and x0=0: l(x)=[phi(x1)-phi(x0)]*x+phi(x0)
 *	We want to expess the l(x) in terms of x0 and x1, so find the weights of x0 and x1!
 *	[phi(x1)-phi(x0)]*x+phi(x0)=t1*phi(x0)+t2*phi(x1)
 *	phi(x1)*x - phi(x0)*x + phi(x0)=t1*phi(x0)+t2*phi(x1)
 *
 *	Then we have:
 *	1) phi(x1)*x = t2*phi(x1)
 *	2) - phi(x0)*x + phi(x0) = t1*phi(x0)
 *
 *	From the 1): t2=x
 *	From the 2):
 *		- phi(x0)*x + phi(x0)
 *		phi(x0)*(-1*x + 1)
 *		t1=(-1*x + 1)=1-x
 *
 *	So: we get the interpolated value ~phi(x)=e(x)=t1*phi(x0)+t2*phi(x1)=(1-x)*phi(x0)+x*phi(x1)
 *
 *	That means, if we want to compute the interpolated value in x (between the point x0 and x1), we have to count
 *	the value at x0 with weight (1-x) and x1 with x.
 *	Weights of x0 and x1:
 *		x0	->	1-x
 *		x1	->	x
 *
 *	The edge endpoint is indexed with 0 (x0) and 1 (x1):
 *		0	->	1-x
 *		1	->	x
 *
 *	A function that returns the edge weight for the endpoints:
 *		w(p, x)	<- -x+1 + p*(2x-1)
 *
 *	That is: w(0,x) returns -x+1 and w(1,x) returns x.
 *
 *	In 2D:
 *		(0,0)	->	(1-x,1-y)
 *		(0,1)	->	(1-x,y)
 *		(1,0)	->	(x,1-y)
 *		(1,1)	->	(x,y)
 *
 *	~phi(x,y) = phi(0,0)*(1-x)*(1-y) + phi(0,1)*(1-x)*y + phi(1,0)*x*(1-y) + phi(1,1)*x*y
 *	So, in 2D we have to compute the area of the squares, and for this reason, we have to
 *	multiply the elements of the resulting pairs.
 *
 *	The linear interpolation can be easily extended to 3D and the w function becomes:
 *		w(p,x)	<-	mul(-x+ONES(3) + hp(p, 2*x-ONES(3))),
 *
 *		where the hp is the Hadamard (pointwise) product of vectors,
 *		and mul multiplies the elemnts of a vector.
 *
 *	Params: cubeVertex=p, cubePoint=x
 */
__device__ __host__ float getVertexWeight(int3 cubeVertex, float3 localCubePt){
	float3 mxp1 = (-1*localCubePt)+FONES3;
	float3 res =
			(mxp1 +
				hp(
						make_float3(cubeVertex),
						((2*localCubePt)-FONES3)
				)
			);
	return mul(res);
}


/*
 * Interpolates the value of a point in 3D based on the function passed in the parameter levelSet.
 * pt: the global coordinate of the point.
 * First, the function determines the reference point ptRef and trilinearly interpolates the values
 * to pt from the eight point refPt+{0,1}^3 with the appropriate weights.
 */
__device__ __host__ float interpolate3D(const float3 pt, const Func3D<float>& func){
	int3 cubeRefPt = make_int3(pt);
	float3 localCubePt = pt-make_float3(cubeRefPt);
	float result = 0.0f;
	for(int i = 0; i <=1; i++){
		for(int j = 0; j <= 1; j++){
			for(int k = 0; k <= 1; k++){
				int3 cubeVertexOff = make_int3(i, j, k);
				float vertexWeight = getVertexWeight(cubeVertexOff, localCubePt);
				result += vertexWeight*func[Point3D(cubeRefPt+cubeVertexOff)];
			}
		}
	}
	return result;
}

__device__ __host__ float3 interpolate3D(const float3 pt, const Func3D<float3>& func){
	int3 cubeRefPt = make_int3(pt);
	float3 localCubePt = pt-make_float3(cubeRefPt);
	float3 result = FZEROS3;
	for(int i = 0; i <=1; i++){
		for(int j = 0; j <= 1; j++){
			for(int k = 0; k <= 1; k++){
				int3 cubeVertexOff = make_int3(i, j, k);
				float vertexWeight = getVertexWeight(cubeVertexOff, localCubePt);
				result += vertexWeight*func[Point3D(cubeRefPt+cubeVertexOff)];
			}
		}
	}
	return result;
}


#endif
