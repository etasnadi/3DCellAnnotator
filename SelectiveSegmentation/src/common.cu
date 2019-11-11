#include "common.cuh"

#include <iostream>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>


string get_file_name_from_path(string path){
	return path.substr(path.find_last_of("/\\") + 1);
}

commonns::Point::Point(){
}

commonns::Point::Point(int a_i, int a_j, int a_k) : i(a_i), j(a_j), k(a_k){
}

commonns::Point::Point(int linearCoord, commonns::Dimensions dim){
	k = linearCoord / (dim.dimX * dim.dimY);
	j = (linearCoord - (k * dim.dimX * dim.dimY)) / (dim.dimX);
	i = (linearCoord - (j * dim.dimX) - (k * dim.dimX * dim.dimY)) / 1;
}

__device__ void getDescartesCoord(int linearCoord, int& i, int& j, int& k, int dimX, int dimY, int dimZ){
	k = linearCoord / (dimX * dimY);
	j = (linearCoord - (k * dimX * dimY)) / (dimX);
	i = (linearCoord - (j * dimX) - (k * dimX * dimY)) / 1;
}

__device__ float3 getFloat3(float* arr, int from){
	return make_float3(arr[from+0], arr[from+1], arr[from+2]);
}

// Returns false if the point is not on the image boundary.
__device__ bool isInnerPoint(int i, int j, int k, int dimX, int dimY, int dimZ){
	return (
			i > 0 && i < dimX-2 &&
			j > 0 && j < dimY-2 &&
			k > 0 && k < dimZ-2);
}

// Returns false if the point is not on the image boundary.
__device__ bool isValidPoint(int i, int j, int k, int dimX, int dimY, int dimZ){
	return (
			i >= 0 && i <= dimX-2 &&
			j >= 0 && j <= dimY-2 &&
			k >= 0 && k <= dimZ-2);
}

// The isValid* functions should be organized later.
// The ones below are correct. I think.
// But its 10:47 PM so should check them tomorrow.
// Or somebody who reads this comment.

// Returns true if (i, j, k) is a valid voxel id.
__device__ bool isValidVoxelId(int i, int j, int k, int dimX, int dimY, int dimZ){
	return (
			i >= 0 && i <= dimX-2 &&
			j >= 0 && j <= dimY-2 &&
			k >= 0 && k <= dimZ-2);
}

// Returns true if the grid point is a valid one.
__device__ bool isValidGridPoint(int i, int j, int k, int dimX, int dimY, int dimZ){
	return (
			i >= 0 && i < dimX &&
			j >= 0 && j < dimY &&
			k >= 0 && k < dimZ);
}

// Returns true if the grid point (i,j,k) identifies an inner voxel.
__device__ bool isInnerVoxelId(int i, int j, int k, int dimX, int dimY, int dimZ){
	return (
			i > 0 && i < dimX-2 &&
			j > 0 && j < dimY-2 &&
			k > 0 && k < dimZ-2);
}

__device__ int sgn(float a){
	if(a < 0){
		return -1;
	}else{
		return 1;
	}
}

// Returns the L1 norm of the vector (i,j,k)
__device__ int getDist(int i, int j, int k){
	return max(max(abs(i), abs(j)), abs(k));
}

__device__ int get_linear_coord(int i, int j, int k, int dimX, int dimY, int dimZ){
	return i + j*dimX + k*dimX*dimY;
}

int commonns::Point::getLinearCoord(commonns::Dimensions d){
	return i + j*d.dimX + k*d.dimX*d.dimY;
}


float3 getFloat3(float a, float b, float c){
	float3 f;
	f.x = a;
	f.y = b;
	f.z = c;
	return f;
}

float4 getFloat4(float a, float b, float c, float d){
	float4 f;
	f.x = a;
	f.y = b;
	f.z = c;
	f.w = d;
	return f;
}

__device__ void printFloat3(float3 p){
	printf("Float3: (%f,%f,%f)", p.x, p.y, p.z);
}

/*
bool commonns::Point::operator==(const commonns::Point &other) {
	return this->i == other.i && this -> j == other.j && this->k == other.k;
}
*/

////////////////////////////////////////////////////////////////////////////////
// Debug
////////////////////////////////////////////////////////////////////////////////

void printData(int* data, char const* filename, int dimX, int dimY, int dimZ)
{
	FILE *file = fopen(filename, "w");
	for(int k = 0; k < dimZ; k++){
		fprintf(file,"%d: \n", k);
		for(int j = 0; j < dimY; j++){
			for(int i = 0; i < dimX; i++){
				fprintf(file, "(%d,%d,%d) %d\t", i, j, k, data[i+j*dimX+k*dimX*dimY]);
			}
			fprintf(file, "\n");
		}
		fprintf(file, "----------------------------------------------\n");
	}
	fclose(file);
}

void printFloatData(float* data, char const* filename, int dimX, int dimY, int dimZ)
{
	FILE *file = fopen(filename, "w");
	for(int k = 0; k < dimZ; k++){
		fprintf(file,"%d: \n", k);
		for(int j = 0; j < dimY; j++){
			for(int i = 0; i < dimX; i++){
				fprintf(file, "(%d,%d,%d) %.5f\t", i, j, k, data[i+j*dimX+k*dimX*dimY]);
			}
			fprintf(file, "\n");
		}
		fprintf(file, "----------------------------------------------\n");
	}
	fclose(file);
}

void copyAndPrint(float *d_data, char const* filename, int dimX, int dimY, int dimZ)
{
	float *h_test;
	h_test = new float[dimX*dimY*dimZ];
	cudaMemcpy(h_test, d_data, sizeof(float)*dimX*dimY*dimZ, cudaMemcpyDeviceToHost);
	printFloatData(h_test, filename, dimX, dimY, dimZ);
	free(h_test);
}

void copyAndPrintInt(int *d_data, char const* filename, int dimX, int dimY, int dimZ)
{
	int *h_test;
	h_test = new int[dimX*dimY*dimZ];
	cudaMemcpy(h_test, d_data, sizeof(int)*dimX*dimY*dimZ, cudaMemcpyDeviceToHost);
	printData(h_test, filename, dimX, dimY, dimZ);
	free(h_test);
}

extern __device__ __host__ int lin(Point3D point, Size3D& size){
	return
			point.x +
			point.y * size.width +
			point.z * size.width*size.height;
}

extern __device__ __host__ bool isValid(Point3D point, Size3D& size){
	bool ret = false;
	if(
			point.x >= 0 && point.x < size.width &&
			point.y >= 0 && point.y < size.height &&
			point.z >= 0 && point.z < size.depth

	){
		ret = true;
	}
	return ret;
}


int Tmax(int a, int b, int c){
	if(a > b){
		if(c > a){
			return c;
		} else {
			return a;
		}
	} else {
		if(c > b){
			return c;
		} else {
			return b;
		}
	}
}

ostream& operator<<(ostream& os, AlgParams& a){
	os << "Alg parameters(lambda, mu, eta, theta): " << a.lambda << "," << a.mu << "," << a.eta << "," << a.theta << endl;
	os << "Alg parameters(preseg eta, preseg theta): " << a.preSegEta << "," << a.preSegTheta << endl;
	os << "Phase parameters(w, norm term, init reg cnt, ev reg cnt): " << a.w << "," << "," << a.regNormTerm << "," << a.initRegCount << "," << a.evolutionRegCount << endl;
	return os;
}

/* Splitting alg: https://stackoverflow.com/questions/236129/split-a-string-in-c */
template<typename Out>
void split(const string &s, char delim, Out result) {
	stringstream ss;
	ss.str(s);
	string item;
	while (std::getline(ss, item, delim)) {
		*(result++) = item;
	}
}

vector<string> split(const string &s, char delim) {
    vector<string> elems;
    ::split(s, delim, std::back_inserter(elems));
    return elems;
}

template<typename T>
struct absolute_value : public unary_function<T,T>
{
  __host__ __device__ T operator()(const T &x) const
  {
    return x < T(0) ? -x : x;
  }
};

float absMax(float *d_data, int size) {
	thrust::device_ptr<float> dptr_d_data(d_data);
	return thrust::transform_reduce(dptr_d_data, dptr_d_data+size, absolute_value<float>(), (float) 0, thrust::maximum<float>());
}

float absMin(float *d_data, int size, float init) {
	thrust::device_ptr<float> dptr_d_data(d_data);
	return thrust::transform_reduce(dptr_d_data, dptr_d_data+size, absolute_value<float>(), init, thrust::minimum<float>());
}
