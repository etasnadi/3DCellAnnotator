#ifndef MARCHING_CUBES_CUH
#define MARCHING_CUBES_CUH 1

#include "cudatools/function.cuh"
#include "common.cuh"

using namespace commonns;

// Shared memory solution. Should be removed in the future!

#define MC_LOC_SIZE 8*sizeof(Point3D)

// To initialise the shared memory from the host code
void callInitSharedMemoryOnDevice();
// Initialises the shared memory for the qw structure
__device__ int initSharedMemory();
extern __shared__ Point3D qw[];

// Device and host tables for marching cubes

extern int edgeConnectionTable[24];
extern int vertexOffsetTable[24];
extern float edgeDirectionTable[36];
extern int cubeEdgeFlags[256];
extern int triangleConnectionTable[4096];

extern __constant__ int d_edgeConnectionTable[24];
extern __constant__ int d_vertexOffsetTable[24];
extern __constant__ float d_edgeDirectionTable[36];
extern __constant__ int d_cubeEdgeFlags[256];
extern __constant__ int d_triangleConnectionTable[4096];

// Marching Cubes implementation

class Cubes{
private:
	float isoValue;
	Func3D<float> func;

	#ifdef __CUDA_ARCH__
		int* edgeConn = d_edgeConnectionTable;
		int* vertexOffset = d_vertexOffsetTable;
		float* edgeDirection = d_edgeDirectionTable;
		int* cubeEdgeFlags = d_cubeEdgeFlags;
		int* triangleConn = d_triangleConnectionTable;
	#else
		int* edgeConn = ::edgeConnectionTable;
		int* vertexOffset = ::vertexOffsetTable;
		float* edgeDirection = ::edgeDirectionTable;
		int* cubeEdgeFlags = ::cubeEdgeFlags;
		int* triangleConn = ::triangleConnectionTable;
	#endif

	__device__ __host__ int cubeOffset(Point3D p, Size3D size, int off);
	__device__ __host__ float getIntersection(float f1, float f2, float fIsolevel);
	__device__ __host__ int computeCubeConfiguration(float* d_data, Point3D p, Size3D size, float isoValue);
	__device__ __host__ void computeIntersectedEdges(float* d_data, Point3D p, Size3D size, float isoValue, int flagIndex, float* edgeIntersection);
	//  Get one triangle in Triangle format using the flagIndex and the edgeIntersection.
	__device__ __host__ Triangle getTriangle(int iTriangle, int flagIndex, float* edgeIntersection);
	// Return the triangles in the cube in Triangle format using the flagIndex and the edgeIntersection
	__device__ __host__ int getCubeTriangles(Triangle* tris, int flagIndex, float* edgeIntersection);
public:
	__device__ __host__ Cubes();
	__device__ __host__ Cubes(Func3D<float> a_func);
	__device__ __host__ Cubes(float a_isoValue, Func3D<float> a_func);
	__device__ __host__ int getNofVoxelTriangles(int3 p);
	__device__ __host__ int getVoxelTriangles(int3 p, Triangle* triangles);
	__device__ __host__ bool onIsoSurface(int3 p);
	// Only can be called from the host!
	static void initGpuTables();
};

#endif
