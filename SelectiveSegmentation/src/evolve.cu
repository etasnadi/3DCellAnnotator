#include <math.h>

#include <boost/log/trivial.hpp>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#include "marchingCubes.cuh"
#include "evolve.cuh"
#include "cudatools/errorHandling.cuh"
#include "localRegion.cuh"
#include "common.cuh"
#include "surfaceAndVolume.cuh"
#include "cudatools/function.cuh"
#include "cudatools/deviceChunk.cuh"

#include "phaseField.cuh"

// Minimim object size
#define MIN_OJB_VOL 10.0f

using namespace std;
using namespace thrust;

// Computing fundamental things

// The sum curvature (k1+k2). The mean curvature is: (k1+k2)/2
__device__ float calcSumK(Derivative3D<float> df) {
	float gn = df.calcGradLen();
	if(gn < 0.1) {
		return 0;
	} else {
		float norm = pow(gn,3);
		return (df.x*df.x*df.yy
			  - 2*df.x*df.y*df.xy
			  + df.y*df.y*df.xx
			  + df.x*df.x*df.zz
			  - 2*df.x*df.z*df.xz
			  + df.z*df.z*df.xx
			  + df.y*df.y*df.zz
			  - 2*df.y*df.z*df.yz
			  + df.z*df.z*df.yy
			   )/norm;
	}
}

__device__ float calcKg(Derivative3D<float> df) {
	float gn = df.calcGradLen();

	if(gn < 0.2) {
		return 0;
	} else {
		float norm = pow(gn,3)+_EPS;
		return  (df.x*df.x*(df.yy*df.zz - df.yz*df.yz)
			   + df.y*df.y*(df.xx*df.zz - df.xz*df.xz)
			   + df.z*df.z*(df.xx*df.yy - df.xy*df.xy)
			   - 2*df.x*df.y*(df.xy*df.zz - df.yz*df.xz)
			   - 2*df.y*df.z*(df.yz*df.xx - df.xy*df.xz)
			   - 2*df.x*df.z*(df.xz*df.yy - df.xy*df.yz)
		   	    )/norm;
	}
}

// Computing the data term

__device__ Point3D closestPt(float3 pt){
	return Point3D(int(lroundf(pt.x)), int(lroundf(pt.y)), int(lroundf(pt.z)));
}

__device__ float dataValueAt(float *d_image, Point3D gridPoint, int3 gridDim, int3 imgDim, float dhGrid) {
	// this function determines the nearest pixel in the image
	// to the given point

	float scaleRatio = (float)imgDim.x/(float)gridDim.x;
	float dhImg = dhGrid/scaleRatio;

	Point3D closestImgPt = closestPt(gridPoint.getFloat3()*scaleRatio);
	FltFunc3D imgFunc(imgDim, d_image);

	return imgFunc.dxx(closestImgPt, dhImg) + imgFunc.dyy(closestImgPt, dhImg) + imgFunc.dzz(closestImgPt, dhImg);
}


__global__ void dataValueTriangle(float *d_dataTerm, float *d_image, float3 *triangles, int nTriangles, int3 gridDim, int3 imDim, float dhGrid) {
	int triangleId = getThread();
	if(triangleId < nTriangles*3) {
		float3 point = triangles[triangleId];
		Point3D closestGridPt = closestPt(point);
		FltFunc3D dataTerm(Size3D(gridDim), d_dataTerm);
		dataTerm[closestGridPt] = dataValueAt(d_image, closestGridPt, gridDim, imDim, dhGrid);
	}
}

void computeDataTerm(float* dataTerm, float* image, Size3D imageSize){
	GpuConf3D conf(imageSize, 4);
	float dhImg = 1.0f;

	/*
		// calculate the laplace field FOR THE IMAGE!
		dchunk_float d_imageLaplace(imageDims.vol());
		cout << "ImageDims: " << imageDims << endl;
		cout << "GridDims: " << gridProps.gridSize << endl;
		cout << "Translation: " << gridToImageTranslation << endl;
		computeL<<<conf_image.grid(), conf_image.block()>>>(d_imageLaplace.getPtr(), d_image.getPtr(), imageDims, dh);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	*/


	computeL<<<conf.grid(), conf.block()>>>(dataTerm, image, imageSize, dhImg);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

__global__ void computeDerivativesKernel(FltDer3D *der, float* image, Size3D imageSize, float dhImg){
	Func3D<FltDer3D> der_f(imageSize, der);
	FltFunc3D image_f(imageSize, image);

	Point3D p = getThread3D();

	if(p > 0 && p < Point3D(imageSize)-1){
		der_f[p] = image_f.der(p, dhImg);
	}
}

void computeImgDerivatives(FltDer3D *der, float* image, Size3D imageSize){
	GpuConf3D conf(imageSize, 4);
	float dhImg = 1.0f;

	computeDerivativesKernel<<<conf.grid(), conf.block()>>>(der, image, imageSize, dhImg);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

// Copmutes the MEAN curvature field.
__global__ void computeK(float *d_result, float *d_fun, float dh, Size3D gridSize) {
	FltFunc3D fun(gridSize, d_fun);
	FltFunc3D K(gridSize, d_result);

	Point3D p = getThread3D();

	if(p > 0 && p < Point3D(gridSize)-1) {
		Derivative3D<float> df = fun.der(p, dh);
		K[p] = calcSumK(df)*0.5;
	}
}

__global__ void computeGradPhi(float3* d_gradField, float* d_fun, float dh, Size3D gridSize){
	Func3D<float> fun(gridSize, d_fun);
	Func3D<float3> _grad(gridSize, d_gradField);

	int3 size = gridSize.geti3();
	int3 p = getThread3D().getInt3();
	if(p > IONES3 && p < size-IONES3){
		_grad[p] = fun.grad(Point3D(p), dh);
	}
}

// Computes the Laplace of the level set. float[]->float[]
__global__ void computeL(float* result, float* func_in, Size3D dims, float dh){
	Point3D p = getThread3D();

	FltFunc3D func(dims, func_in);
	FltFunc3D L(dims, result);

	if(p > 0 && p < Point3D(dims)-1){
		L[p] = func.dxx(p, dh) + func.dyy(p, dh) + func.dzz(p, dh);
	}else if(p >= 0 && p <= Point3D(dims)-1){
		L[p] = 0.0f;
	}
}

// Computes the normal field. Handy if we need the divergence of the unit normal: e.g.:
// Geodesic AC or it is needed in the Local region as well. float->float3[]
__global__ void computeN(float3 *d_result, float *d_fun, float dh, Size3D gridSize) {
	FltFunc3D fun(gridSize, d_fun);
	Func3D<float3> N(gridSize, d_result);

	Point3D p = getThread3D();

	if(p > 0 && p < Point3D(gridSize)-1) {
		float3 n = fun.normal(p, dh);
		N[p] = n;
	}
}

__device__ float _cDiff(float valp, float valm, float dh){
	return (valp - valm)/2*dh;
}

__device__ float _div(Point3D p, Func3D<float3>& vec_fun){
	FMat NNab(3, 3);

	// Unit normals in the neighbouring points
	float3 g_xm = vec_fun[p+Point3D(-1, 0, 0)];
	float3 g_xp = vec_fun[p+Point3D(1, 0, 0)];

	float3 g_ym = vec_fun[p+Point3D(0, -1, 0)];
	float3 g_yp = vec_fun[p+Point3D(0, 1, 0)];

	float3 g_zm = vec_fun[p+Point3D(0, 0, -1)];
	float3 g_zp = vec_fun[p+Point3D(0, 0, 1)];

	// n1
	float xx = _cDiff(g_xp.x, g_xm.x, 1.0f);

	// n2
	float yy = _cDiff(g_yp.y, g_ym.y, 1.0f);

	// n3
	float zz = _cDiff(g_zp.z, g_zm.z, 1.0f);

	return xx + yy + zz;
}

// Computes the divergence of the vector field.
__global__ void computeDiv(float* d_result, float3* d_fun, float dh, Size3D gridSize){
	FltFunc3D result(gridSize, d_result);
	Func3D<float3> fun(gridSize, d_fun);

	Point3D p = getThread3D();

	if(p > ONES3 && p < Point3D(gridSize)-(2)) {
		result[p] = _div(p, fun);
	}

}

__device__ float calcTrace(Derivative3D<float> df, Derivative3D<float> dK){
	return dK.xx + dK.yy + dK.zz
	- (  dK.xx*df.x*df.x + dK.xy*df.x*df.y + dK.xz*df.x*df.z
			+ dK.xy*df.x*df.y + dK.yy*df.y*df.y + dK.yz*df.y*df.z
			+ dK.xz*df.x*df.z + dK.yz*df.y*df.z + dK.zz*df.z*df.z
	   )/(pow(df.calcGradLen(),2)+_EPS);
}

typedef struct {
	float *lr;
} instrument;

__device__ float3 calcS0t(Func3D<float> levelSet, int3 p, float3 objCog){
	float3 s0t = make_float3(0.0, 0.0, 0.0);
	Cubes marchingCubes(levelSet);
	Triangle tris[5];
	int nTris = marchingCubes.getVoxelTriangles(p, tris);
	for(int i = 0; i < nTris; i++){
		Triangle tri = tris[i];
		s0t = s0t + (tri.cog()-objCog) / float(nTris);
	}
	return s0t;
}



__global__ void computeSpeed(float *d_speed, float *d_out, float *d_in, float *d_K, float* d_div_in, float *d_image, float *d_dataTerm, FltDer3D* imgDers, int3 gridToImageTranslation, int *d_neighs,
								 CurveProps& curParams, float* surfs, float* vols, float3* cogs, float* sms,
								 float dh,
								 GridParams gProps, Size3D imageDims, AlgParams aParams, Obj pref, instrument ins)
{
	Point3D p = getThread3D();
	int3 pImg = p.getInt3()-gridToImageTranslation;

	float prefP = pref.p;
	float prefQ = pref.q;
	float prefVol = pref.vol;
	float prefSurf = pref.surf;

	IntFunc3D neighs(gProps.gridSize, d_neighs);
	FltFunc3D speed(gProps.gridSize, d_speed);
	FltFunc3D f_in(gProps.gridSize, d_in);
	FltFunc3D f_out(gProps.gridSize, d_out);
	FltFunc3D K(gProps.gridSize, d_K);
	FltFunc3D f_ins_lr(imageDims, ins.lr);
	FltFunc3D f_image(imageDims, d_image);
	FltFunc3D f_div(gProps.gridSize, d_div_in);
	FltFunc3D f_data(imageDims, d_dataTerm);


	float minObjVol = MIN_OJB_VOL;

	if(p > IONES3 && p < (gProps.gridSize.geti3()-(2*IONES3))){
		int objectID = neighs[p];
		int realObejctId = objectID + 1;

		if(objectID == -1){ // No object associated with this grid point, the velocity should be zero
			speed[p] = 0.0f;
		} else {
			// Object properties
			float objectSurf = surfs[realObejctId];
			float objectVol = vols[realObejctId];
			float3 objectCoG = cogs[realObejctId]/objectSurf;
			float objectM2 = sms[realObejctId];

			if(objectVol < 0.0f) {
				speed[p] = -200;
			} else if(objectVol < minObjVol){
				speed[p] = 10;
			} else {
				Derivative3D<float> df = f_in.der(p, dh);
				Derivative3D<float> dK = K.der(p, dh);
				float3 f_grad = df.grad();
				float Khalf = 0.5*K[p];			// The half of the curvature
				float f_gradLen = df.calcGradLen();	// The L2 norm of the gradient of the level set
				float f_gradn_sq = pow(f_gradLen, 2);

				float dataVal = 0.0f;

				// Data term
				if(aParams.dataTerm == aParams.DATA_TERM_LAPLACE){
					// 1.) Edge based (Laplace)

					// Compute on the fly
					// Derivative3D<float> df_im = f_image.der(p, dh);
					// dataVal = df_im.xx + df_im.yy +df_im.zz;

					if(pImg < f_data.getSize().geti3() && pImg > IZEROS3){
						dataVal = f_data[pImg];
					}
				}else if(aParams.dataTerm == aParams.DATA_TERM_LOCAL_INTEGRAL){
					FMat NNab = calcNormNab(f_in, p);
					dataVal = calcLR_(d_image, imageDims.geti3(), imgDers, gProps, pImg, d_in, f_grad, NNab, aParams);
				}

				float Kg = calcKg(df);		// The gaussian curvature of the level set

				// Volume prior
				//float vol = f_gradn_sq*pow(objectVol-prefVol,2)/(2*pow(prefVol,3));
				float vol = f_gradn_sq*(objectVol-prefVol)/(2*pow(prefVol,2));

				// Plasma prior
				float plasma = -f_gradn_sq
										*(pow(objectSurf, 3.0f/2.0f)-prefP*objectVol)
										*(prefP-3*Khalf*sqrt(objectSurf))/(prefVol);

				// Ellipsoid prior (does not work yet)
				float3 unitNorm = f_in.normal(p, dh);
				// <-- Solution #1

				float _prefQ = 1.6f;

				float3 s0t = calcS0t(f_in, p.getInt3(), objectCoG);
				float s0tsq = s0t*s0t;

				// Ellipsoid prior equation #1
				float ellipsoid1 = -2*f_gradn_sq*
						(pow(objectSurf,2)-(prefQ*objectM2))*
						(K[p]*0.5*(prefQ*s0tsq-2*objectSurf)-prefQ*(s0t*unitNorm))/(pow(prefSurf,4) );

				// Ellipsoid prior equation #2
				float vol23 = pow(objectVol, float(2.0f/3.0f));
				float volm13 = pow(objectVol, -float(1.0f/3.0f));
				float ellipsoid2 =
						-2*f_gradn_sq*
						(vol23*objectSurf - _prefQ*objectM2)*
						(-1*Khalf*vol23 - float(1.0f/3.0f)*volm13*objectSurf - _prefQ*(s0t*unitNorm - s0tsq*Khalf));

				// -->

				// Euler elastica
				float trace = calcTrace(df, dK);
				float smooth = -(0.5*pow(K[p],3)
						 	   - 2*K[p]*Kg
						 	   + trace)*f_gradn_sq;

				// Putting everything together
				if(aParams.model == aParams.MODEL_SELECTIVE){
					speed[p] =  aParams.lambda*vol
								+ aParams.mu*plasma
								+ aParams.eta*f_gradLen*dataVal
								+ aParams.theta*smooth;
				}else if(aParams.model == aParams.MODEL_CLASSIC){
					speed[p] =
							aParams.eta*f_gradLen*dataVal +
							aParams.theta*f_gradLen*Khalf;// + 0.01*f_gradn;
				}
			}
		}
	}
}

__global__ void evolve(float *d_out, float *d_in, float *d_speedFunction, Size3D gridSize, float dt){
	Point3D p = getThread3D();
	
	FltFunc3D speedFunc(gridSize, d_speedFunction);
	FltFunc3D f_in(gridSize, d_in);
	FltFunc3D f_out(gridSize, d_out);

	if(p > 1 && p < Point3D(gridSize)-2){
		f_out[p] = f_in[p] + dt*speedFunc[p];
	}
}

void launchEvolve(dchunk_float& d_out, dchunk_float& d_in, dchunk_float& d_image, float* d_dataTerm, FltDer3D* imgDers, int3 gridToImageTranslation,
				dchunk<float3>& normals, dchunk<float>& K,
				int *d_nodeID,
				CurveProps& curParams,
                GridParams gridProps, Size3D imageDims,
                AlgParams aParams,
                int iter,
                Obj pref	/* preferred shape */)
{ 
	int nVoxels = gridProps.gridSize.vol();
	float dh = gridProps.gridRes;
	GpuConf3D conf_grid(gridProps.gridSize, 4);
	GpuConf3D conf_image(imageDims, 4);

	dchunk_float lr_deb(gridProps.gridSize.vol());
	thrust::fill(lr_deb.getTPtr(), lr_deb.getTPtr()+lr_deb.getElements(), -11111.0f);
	instrument ins;
	ins.lr = lr_deb.getPtr();
	computeK<<<conf_grid.grid(), conf_grid.block()>>>(K.getPtr(), d_in.getPtr(), dh, gridProps.gridSize); // Mean curvature field
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// compute the gradient field
	dchunk<float3> d_gradField(gridProps.gridSize.vol());
	computeGradPhi<<<conf_grid.grid(), conf_grid.block()>>>(d_gradField.getPtr(), d_in.getPtr(), dh, gridProps.gridSize);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// calculate the normal field
	//dchunk<float3> d_n(gridProps.gridSize.vol());
	computeN<<<conf_grid.grid(), conf_grid.block()>>>(normals.getPtr(), d_in.getPtr(), dh, gridProps.gridSize);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// calculate the divergence of the gradient
	dchunk<float> d_div(gridProps.gridSize.vol());
	computeDiv<<<conf_grid.grid(), conf_grid.block()>>>(d_div.getPtr(), normals.getPtr(), dh, gridProps.gridSize);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// compute the speed function
	DevFloatChk d_speed(gridProps.gridSize.vol());
	d_speed.fill(0.0f);
	computeSpeed<<<conf_grid.grid(), conf_grid.block()>>>(d_speed.getPtr(), d_out.getPtr(), d_in.getPtr(), K.getPtr(), d_div.getPtr(), d_image.getPtr(), d_dataTerm, imgDers, gridToImageTranslation, d_nodeID,
												  curParams, curParams.surfContribs.second->getPtr(), curParams.volContribs.second->getPtr(), curParams.cogContribs.second->getPtr(), curParams.smContribs.second->getPtr(),
												  dh, gridProps, imageDims,
												  aParams,
												  pref, ins);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	float dt = 0.05;
	float maxTimeDel = 2.0f/aParams.w;
	float maxVelo = absMax(d_speed.getPtr(), nVoxels);
	//maxTimeDel = 1.0;
	if(maxVelo > 0.00001){
		if(aParams.selective) {
			BOOST_LOG_TRIVIAL(info) << "Segmentation mode: selective.";
			dt = maxTimeDel*dh/maxVelo;
		}else{
			BOOST_LOG_TRIVIAL(info) << "Segmentation mode: regular.";
		}
		BOOST_LOG_TRIVIAL(info) << "Max. velocity: " << absMax(d_speed.getPtr(), nVoxels) << ", timestep: " << dt;
		BOOST_LOG_TRIVIAL(info) << "Max. gradnorm: " << absMax(d_out.getPtr(), nVoxels) << ", min. gradnorm: " << absMin(d_out.getPtr(), nVoxels, 1000);

		// update the level set by the speed function
		evolve<<<conf_grid.grid(), conf_grid.block()>>>(d_out.getPtr(), d_in.getPtr(), d_speed.getPtr(), gridProps.gridSize, dt);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}else{
		BOOST_LOG_TRIVIAL(info) << "Warning: velocity field is (close to) zero!";
	}
}


