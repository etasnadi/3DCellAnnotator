//#include <boost/graph/adjacency_list.hpp>

#ifndef COMMON_CUH
#define COMMON_CUH 1

#include <stdio.h>
#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <iterator>

#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>

#include "cudatools/deviceChunk.cuh"
#include "cudatools/cudatools.cuh"

#include "SimpleConfig.cuh"

#define _COEFF 1.0f

// The object (foreground) intensity in the masks.
#define INT_OBJ 255
// The background intensity in the masks.
#define INT_BACK 0

#define TRUE 1
#define FALSE 0

#define DEBUG 1

#ifdef DEBUG
#define dout cout
#define dprintf printf
#else
#define dout 0 && cout
#define dprintf 0 && printf
#endif

#define SQ_TPB_3D 4
#define SQ_TPB_2D 8
#define TPB_1D 64

#define LIN_IDX i+j*dimX+k*dimX*dimY

#ifdef _WIN32
#define DIRECTORY_SEPARATOR \\
#elif __linux__
#define DIRECTORY_SEPARATOR /
#else
#define DIRECTORY_SEPARATOR /
#endif

using namespace std;

string get_file_name_from_path(string path);

typedef struct {
	int *p_edges;
	int numOfEdges;
} graph;

template <typename T>
__device__ int d_sgn(T val){
	if(val < 0){
		return -1;
	}else{
		return 1;
	}
}

class Triangle {
public:
	float3 A;
	float3 B;
	float3 C;

	/*
	__device__ __host__ Triangle(const Triangle& t){
		A = t.A;
		A = t.B;
		A = t.C;
	}
	 */

	__device__ __host__ Triangle(float3 a_A, float3 a_B, float3 a_C){
		A = a_A;
		A = a_B;
		A = a_C;
	}
	__device__ __host__ Triangle() : Triangle(make_float3(.0, .0, .0), make_float3(.0, .0, .0), make_float3(.0, .0, .0)) {};
	__device__ __host__ float3 cog() const {
		return (A+B+C) * float(1.0f/3.0f);
	}

	__device__ __host__ float area() const {
		float3 amb = A-B;
		float3 cmb = C-B;
		float3 amc = A-C;
		float a = L2(amb);
		float b = L2(cmb);
		float c = L2(amc);
		float ar = 0.0f;
		if(!::isnan(0.25*sqrtf(fabsf((a+b+c)*(-a+b+c)*(a-b+c)*(a+b-c))))){
			ar =  0.25*sqrtf(fabsf((a+b+c)*(-a+b+c)*(a-b+c)*(a+b-c)));
		}
		return ar;
	}

	__device__ __host__ float perim() const {
		return L2(B-A) + L2(B-C) + L2(C-A);
	}

	__device__ __host__ Triangle getScaled(float factor) const {
		Triangle t(A*factor, B*factor, C*factor);
		return t;
	}
};

__device__ int sgn(float a);

// Returns false if the point is not on the image boundary.
__device__ bool isInnerPoint(int i, int j, int k, int dimX, int dimY, int dimZ);

// Returns true if the point is a valid voxel id.
__device__ bool isValidPoint(int i, int j, int k, int dimX, int dimY, int dimZ);

// Returns false if the point is not on the image boundary.
extern __device__ bool isValidVoxelId(int i, int j, int k, int dimX, int dimY, int dimZ);

// Returns false if the point is not on the image boundary.
extern __device__ bool isValidGridPoint(int i, int j, int k, int dimX, int dimY, int dimZ);

__device__ bool isInnerVoxelId(int i, int j, int k, int dimX, int dimY, int dimZ);

extern __device__ int getDist(int i, int j, int k);

int divUp(int a, int b);

////////////////////////////////////////////////////////////////////////////////
// Lookup tables
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// cube vertex indexing convention
//
//             4--------5
//            /|       /|
//           / |      / |
//          /  |     /  |
//         7--------6   |
//         |   0----|---1
//         |  /     |  /
//         | /      | /
//         |/       |/
//         3--------2
//
////////////////////////////////////////////////////////////////////////////////
// h_vertexOffsetTable[vertex][coordinate] = h_vertexOffsetTable[3*vertex + coordinate], coordinate = 0,1,2; vertex = 0,1,...,7
// __constant__ int d_vertexOffsetTable[24];

////////////////////////////////////////////////////////////////////////////////
// cube neighbor indexing convention
////////////////////////////////////////////////////////////////////////////////
//
//			4
//			|  0
//			| /
//			|/
//	  3-----o-----1
//		   /|
//		  / |
//		 2  |
//			5
//
////////////////////////////////////////////////////////////////////////////////
// h_cubeNeighborTable[vertex][coordinate] = h_cubeNeighborTable[3*vertex + coordinate], coordinate = 0,1,2; vertex = 0,1,...,6
// __constant__ int d_cubeNeighborTable[9];

namespace commonns{

	typedef struct {
		int i, j, k;
	} pt;

	typedef struct {
		int X, Y, Z;
	} dim;

	class Point{
	public:
		int i, j, k;
		Point();
		Point(int a_i, int a_j, int a_k);
		Point(int linearCoord, Dimensions dim);
		int getLinearCoord(Dimensions d);
		//bool operator==(Point& other);
	};
}

__device__ void getDescartesCoord(int linearCoord, int& i, int& j, int& k, int dimX, int dimY, int dimZ);

__device__ int get_linear_coord(int i, int j, int k, int dimX, int dimY, int dimZ);

__device__ float3 getFloat3(float* arr, int from);

__device__ void printFloat3(float3 p);

float3 getFloat3(float a, float b, float c);
float4 getFloat4(float a, float b, float c, float d);

void printData(int* data, char const* filename, int dimX, int dimY, int dimZ);
void printFloatData(float* data, char const* filename, int dimX, int dimY, int dimZ);
void copyAndPrint(float *d_data, char const* filename, int dimX, int dimY, int dimZ);
void copyAndPrintInt(int *d_data, char const* filename, int dimX, int dimY, int dimZ);

// Size 3D impl

extern __device__ __host__ int lin(Point3D point, Size3D& size);
extern __device__ __host__ bool isValid(Point3D point, Size3D& size);

int Tmax(int a, int b, int c);

// Only the p and vol is used. The q is for the ellipsoid prior and surf is for the surface prior that are not implemented yet.
typedef struct {
	float p;
	float vol;

	float q;
	float surf;

	void updateFromConfig(SimpleConfig& a_conf){
		if(a_conf.isSetProperty("pref.p")){
			p = a_conf.getFProperty("pref.p");
		}

		if(a_conf.isSetProperty("pref.vol")){
			vol = a_conf.getFProperty("pref.vol");
		}
	}

} Obj;

#define DATA_TERM_LAPLACE_VAL "LAP"
#define DATA_TERM_LOCAL_INTEGRAL_VAL "LR"

#define MODEL_CLASSIC_VAL "CLASSIC"
#define MODEL_SELECTIVE_VAL "SEL"


typedef struct {
	// 0: Edge based (laplace)
	// 1: Local integral
	int DATA_TERM_LAPLACE = 0;
	int DATA_TERM_LOCAL_INTEGRAL = 1;

	int dataTerm = 0;

	int MODEL_CLASSIC = 0;
	int MODEL_SELECTIVE = 1;

	int model = 0;

	// Selective parameters
	float lambda;
	float mu;
	float eta;
	float theta;

	// Pre segmentation parameters
	float preSegEta;
	float preSegTheta;

	int3 regionExtent;

	// Phase field equation parameters
	float w = 4.0f; // Zero crossing width
	int initRegCount = 50;
	int evolutionRegCount = 20;
	float regNormTerm = 0.01;

	// Initialisation
	float3 sphere_center;
	float sphere_r;

	bool selective = true;

	void updateFromConfig(SimpleConfig& a_conf){
		if(a_conf.isSetProperty("eq.mu")){
			mu = a_conf.getFProperty("eq.mu");
		}

		if(a_conf.isSetProperty("eq.lambda")){
			lambda = a_conf.getFProperty("eq.lambda");
		}

		if(a_conf.isSetProperty("eq.eta")){
			eta = a_conf.getFProperty("eq.eta");
		}

		if(a_conf.isSetProperty("eq.theta")){
			theta = a_conf.getFProperty("eq.theta");
		}

		// Disable or enable the parameters

		if(a_conf.isSetProperty("gui.eq.mu.enabled")){
			mu *= a_conf.getIProperty("gui.eq.mu.enabled");
		}

		if(a_conf.isSetProperty("gui.eq.lambda.enabled")){
			lambda *= a_conf.getIProperty("gui.eq.lambda.enabled");
		}

		if(a_conf.isSetProperty("gui.eq.eta.enabled")){
			eta *= a_conf.getIProperty("gui.eq.eta.enabled");
		}

		if(a_conf.isSetProperty("gui.eq.theta.enabled")){
			theta *= a_conf.getIProperty("gui.eq.theta.enabled");
		}

		//BOOST_LOG_TRIVIAL(info) << "Parameters reveived: " << a_conf;

		BOOST_LOG_TRIVIAL(info) << "Updated params: mu: " << mu << " lambda: " << lambda << " eta: " << eta << " theta: " << theta;
	}
}AlgParams;

ostream& operator<<(ostream& os, AlgParams& a);

typedef struct {
	int gridRes;
	Size3D gridSize;
} GridParams;

/* Splitting alg: https://stackoverflow.com/questions/236129/split-a-string-in-c */
template<typename Out>
void split(const string &s, char delim, Out result);
vector<string> split(const string &s, char delim);


float absMax(float *d_data, int size);
float absMin(float *d_data, int size, float init);

#endif
