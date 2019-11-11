#include "matrix.cuh"

template<>
__device__ __host__ void Mat<float>::s(int i, int j, float val){
	d[i*w+j] = val;
}

template<>
__device__ __host__ float Mat<float>::e(int i, int j){
	return d[i*w+j];
}

// Init with spec vals
template<>
__device__ __host__ void Mat<float>::initv(float val,int aH, int aW){
	w = aW;
	h = aH;
	for(int i = 0; i < w*h; i++){
		d[i] = val;
	}
}

// Init matrix with zeros
template<>
__device__ __host__ Mat<float>::Mat(int aH, int aW){
	initv(0.0f, aH, aW);
}

template<>
__device__ __host__ Mat<float>::Mat(FltDer3D df){
	initv(0.0f, 3, 3);

	s(0,0,df.xx);
	s(0,1,df.xy);
	s(0,2,df.xz);

	s(1,0,df.xy);
	s(1,1,df.yy);
	s(1,2,df.yz);

	s(2,0,df.xz);
	s(2,1,df.yz);
	s(2,2,df.zz);
}


template<>
__device__ __host__ Mat<float> Mat<float>::cvec3(float3 f){
	Mat<float> q(3,1);
	q.s(0,0,f.x);
	q.s(1,0,f.y);
	q.s(2,0,f.z);
	return q;
}

template<>
__device__ __host__ Mat<float> Mat<float>::rvec3(float3 f){
	Mat<float> q(1,3);
	q.s(0,0,f.x);
	q.s(0,1,f.y);
	q.s(0,2,f.z);
	return q;
}

// Get a col vec component from (0,0)
template<>
__device__ __host__ float3 Mat<float>::getRvec3(){
	return make_float3(e(0,0), e(1,0), e(2,0));
}

// Get a col vec component from (0,0)
template<>
__device__ __host__ float3 Mat<float>::getCvec3(){
	return make_float3(e(0,0), e(0,1), e(0,1));
}

__device__ __host__ Mat<float> eye(int wh){
	return eye(wh, wh);
}

// Identity
__device__ __host__ Mat<float> eye(int h, int w){
	Mat<float> ret(h, w);

	for(int i = 0; i < min(ret.h,ret.w); i++){
		ret.s(i,i,1);
	}

	return ret;
}

template<>
__device__ __host__ Mat<float> Mat<float>::trans(){
	Mat<float> ret(w, h);
	for(int i = 0; i < h; i++){
		for(int j = 0; j < w; j++){
			ret.s(j, i, e(i, j));
		}
	}
	return ret;
}

template<>
__device__ __host__ Mat<float>::operator float(){
	if(w != 1 || h!= 1){
		printf("Attempted to cast to scalar but the actual size of the matrix is %d x %d", h, w);
	}
	return e(0,0);
}

template<>
__device__ __host__ void Mat<float>::pr(){
	printf("Matrix: %d x %d\n", h, w);
	for(int i = 0; i < h; i++){
		for(int j = 0; j < w; j++){
			printf("%f ", e(i, j));
		}
		printf("\n");
	}
	printf("\n");
}

// Mat + Mat
__device__ __host__ Mat<float> operator+(Mat<float> mat1, Mat<float> mat2){
	Mat<float> ret(mat1.h, mat1.w);

	for(int i = 0; i < mat1.h; i++){
		for(int j = 0; j < mat1.w; j++){
			float sum = mat1.e(i,j)+mat2.e(i,j);
			ret.s(i, j, sum);
		}
	}

	return ret;
}

// Mat - Mat
__device__ __host__ Mat<float> operator-(Mat<float> mat1, Mat<float> mat2){
	Mat<float> ret(mat1.h, mat1.w);

	for(int i = 0; i < mat1.h; i++){
		for(int j = 0; j < mat1.w; j++){
			float sum = mat1.e(i,j)-mat2.e(i,j);
			ret.s(i, j, sum);
		}
	}

	return ret;
}


// Mat*Mat (m x n)*(n x k)=(m x k)
__device__ __host__ Mat<float> operator*(Mat<float> mat1, Mat<float> mat2){
	Mat<float> ret(mat1.h, mat2.w);

	if(mat1.w != mat2.h){
		printf("Error in mat*mat.\n");
	}

	for(int i = 0; i < mat1.h; i++){
		for(int j = 0; j < mat1.w; j++){
			float sum  = 0;
			for(int k = 0; k < mat1.w; k++){
				sum += float(mat1.e(i,k))*float(mat2.e(k,j));
			}
			ret.s(i, j, sum);
		}
	}

	return ret;
}

// Mat*f
__device__ __host__ Mat<float> operator*(Mat<float> mat, float c){
	Mat<float> ret(mat.h, mat.w);

	for(int i = 0; i < mat.h; i++){
		for(int j = 0; j < mat.w; j++){
			ret.s(i, j, mat.e(i,j)*c);
		}
	}

	return ret;
}

// f*Mat3
__device__ __host__ Mat<float> operator*(float c, Mat<float> mat){
	Mat<float> ret(mat.h, mat.w);

	for(int i = 0; i < mat.h; i++){
		for(int j = 0; j < mat.w; j++){
			ret.s(i, j, mat.e(i,j)*c);
		}
	}

	return ret;
}

// f3*Mat
__device__ __host__ Mat<float> operator*(float3 rvec, Mat<float> mat){
	Mat<float> rvecmat = Mat<float>::rvec3(rvec);

	if(mat.h != 3){
		printf("Error in f3*mat.\n");
	}

	return rvecmat*mat;
}

// Mat*f3
__device__ __host__ Mat<float> operator*(Mat<float> mat, float3 cvec){
	Mat<float> cvecmat = Mat<float>::cvec3(cvec);

	if(mat.h != 3){
		printf("Error in mat*f3.\n");
	}

	return mat*cvecmat;
}

__device__ __host__ float MdotR(Mat<float> mat1, Mat<float> mat2){
	return mat1.e(0,0)*mat2.e(0,0) + mat1.e(0,1)*mat2.e(0,1) + mat1.e(0,2)*mat2.e(0,2);
}

template<>
__device__ __host__ float Mat<float>::tr(){
	float ret = 0.0f;
	for(int i = 0; i < min(w, h); i++){
		ret += e(i,i);
	}
	return ret;
}
