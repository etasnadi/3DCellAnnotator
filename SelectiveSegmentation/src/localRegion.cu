#include "localRegion.cuh"
#include "common.cuh"
#include "matrix.cuh"
#include "marchingCubes.cuh"

__device__ float3 unitNormP(FltFunc3D& f_in, Point3D p){
	float dh = 1.0f;
	float3 gradP = f_in.grad(p, dh);
	float3 uNorm = gradP/(L2(gradP) + _EPS);
	return uNorm;
}

__device__ float cDiff(float valp, float valm, float dh){
	return (valp - valm)/2*dh;
}

__device__ FMat calcNormNab(FltFunc3D& f_in, Point3D p){
	FMat NNab(3, 3);

	// Unit normals in the neighbouring points
	float3 g_xm = unitNormP(f_in, p+Point3D(-1, 0, 0));
	float3 g_xp = unitNormP(f_in, p+Point3D(1, 0, 0));

	float3 g_ym = unitNormP(f_in, p+Point3D(0, -1, 0));
	float3 g_yp = unitNormP(f_in, p+Point3D(0, 1, 0));

	float3 g_zm = unitNormP(f_in, p+Point3D(0, 0, -1));
	float3 g_zp = unitNormP(f_in, p+Point3D(0, 0, 1));

	// n1
	NNab.s(0,0, cDiff(g_xp.x, g_xm.x, 1.0f));
	NNab.s(0,1, cDiff(g_yp.x, g_ym.x, 1.0f));
	NNab.s(0,2, cDiff(g_zp.x, g_zm.x, 1.0f));

	// n2
	NNab.s(1,0, cDiff(g_xp.y, g_xm.y, 1.0f));
	NNab.s(1,1, cDiff(g_yp.y, g_ym.y, 1.0f));
	NNab.s(1,2, cDiff(g_zp.y, g_zm.y, 1.0f));

	// n3
	NNab.s(2,0, cDiff(g_xp.z, g_xm.z, 1.0f));
	NNab.s(2,1, cDiff(g_yp.z, g_ym.z, 1.0f));
	NNab.s(2,2, cDiff(g_zp.z, g_zm.z, 1.0f));

	return NNab;
}


__device__ FMat calcGNormNab(FltFunc3D& f_in, Point3D p){
	FMat NNab(3, 3);

	// Unit normals in the neighbouring points
	float3 g_xm = unitNormP(f_in, p+Point3D(-1, 0, 0));
	float3 g_xp = unitNormP(f_in, p+Point3D(1, 0, 0));

	float3 g_ym = unitNormP(f_in, p+Point3D(0, -1, 0));
	float3 g_yp = unitNormP(f_in, p+Point3D(0, 1, 0));

	float3 g_zm = unitNormP(f_in, p+Point3D(0, 0, -1));
	float3 g_zp = unitNormP(f_in, p+Point3D(0, 0, 1));

	// n1
	NNab.s(0,0, cDiff(g_xp.x, g_xm.x, 1.0f));

	// n2
	NNab.s(1,1, cDiff(g_yp.y, g_ym.y, 1.0f));

	// n3
	NNab.s(2,2, cDiff(g_zp.z, g_zm.z, 1.0f));

	return NNab;
}


/*

Computes the value for the local region.
d_image: the input image,
d_ders: the first and second order derivatives of the image
S: the position where we want to compute the data value
dl_level: the level set
grad: the gradient of the level set at the point S
NNab: the divergence of the gradient at the point S

*/
// p prefix: (point) in the phase field coordinate system
// i prefix: (point) in the image coordinate system
// pl prefix: (point) in the phase field local coordinate system with origin pS
// il prefix: (point) in the image local coordinate system with origin pS*scale, scale = image_x/grid_x
__device__ float calcLR_(
		float* d_image,
		int3 imgDims,
		FltDer3D* d_ders,
		GridParams gProps,
		Point3D S,
		float* d_level,
		float3 grad,
		FMat NNab,
		AlgParams aParams){

	int scal = gProps.gridRes;
	Size3D imgSize = Size3D(imgDims);

	FltFunc3D im(imgSize, d_image);
	Func3D<FltDer3D> ders(imgSize, d_ders);

	// compute pn, pe1, pe2
	float3 n = grad/L2(grad);
	float3 e1 = cross(make_float3(0, 1, 0), n);
	e1 = e1 / (L2(e1) + _EPS);
	float3 e2 = cross(n, e1);

	Point3D lr(aParams.regionExtent.x, aParams.regionExtent.y, aParams.regionExtent.z);

	// xi	e1	lrSiz.x
	// zeta	e2	lrSiz.y
	// eta	n	lrSiz.z

	const int nTerms = 4;
	float terms[nTerms][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
	const int Rm = 0;
	const int Rp = 1;

	for(int xi = -lr.x; xi <= lr.x; xi++){
		for(int zeta = -lr.y; zeta <= lr.y; zeta++){
			for(int eta = -lr.z; eta <= lr.z; eta++){
				if(eta == 0){
					continue;
				}

				int sid = (eta>0) ? Rp : Rm; // if positive side -> side = 1, negative: side = 0

				float3 Ploc = make_float3(float(xi), float(zeta), float(eta));
				Point3D Pim = (Ploc.x*e1) + (Ploc.y*e2) + (Ploc.z*n) + S; // float3 + Point3D. The float3 will be converted to Point3D by trimming.

				// If Pim is outside
				Pim = Pim*scal;

				if(Pim > 0 && Pim < imgSize){
					float3 imGrad = ders[Pim].grad();

					// Term 1
					terms[0][sid] += imGrad*n;

					// Term 2
					FMat ce1 = FMat::cvec3(e1);
					FMat ce2 = FMat::cvec3(e2);
					FMat cn = FMat::cvec3(n);

					FMat re1 = FMat::rvec3(e1);
					FMat re2 = FMat::rvec3(e2);
					FMat rn = FMat::rvec3(n);

					FMat H = ders[Pim];
					terms[1][sid] += float(eta)*re1*H*ce1 
						- float(xi)*rn*H*ce1 
						+ float(eta)*re2*H*ce2 
						- float(zeta)*rn*H*ce2;


					// Term 3
					FMat cnu = NNab*ce1;
					FMat cnv = NNab*ce2;

					FMat rgrad = FMat::rvec3(imGrad);

					FMat l(float(eta)*cnu - xi*(float(rn*ce1))*cnu - float(zeta)*(float(rn*ce2))*cnu);
					float r1((float(eta)*re1-float(xi)*rn)*H*l);
					float r2((float(eta)*re2-float(zeta)*rn)*H*l);
					float r3((float(xi)*rgrad*cnu) - (float(zeta)*rgrad*cnv));
					float r4(float(eta)*(rgrad*cn)*NNab.tr());

					terms[2][sid] += r1 + r2 -r3 -r4;

					// Term4
					float FI = im[Pim];
					terms[3][sid] += FI*NNab.tr();
				}else{
					return 0.0f;
				}
			}
		}
	}

	float norm = 1.0f/4*float(2*((lr.x)*(lr.y)*(lr.z))-lr.z);
	float dataVal = 0.0f;
	for(int term = 0; term < nTerms; term++){
		dataVal += norm*(terms[term][Rp] - terms[term][Rm]);
	}

	return dataVal;
}

__device__ lr_basis getLrBasis(float3 grad){
	lr_basis basis;
	basis.n = grad/L2(grad);
	basis.e1 = cross(make_float3(0, 1, 0), basis.n);
	basis.e1 = basis.e1 / (L2(basis.e1) + _EPS);
	basis.e2 = cross(basis.n, basis.e1);
	return basis;
}

/*
 * Returns the point in the local region.
 * PointLoc: the point in the coordinate sysytem of the local region
 * lr_basis: contains the three basis vectors that defines the orientation of the local region
 * S: the point in the embedding coordinate system (the point of the surface that is the origin
 * 	of the local region after the embedding)
 * scal: the number of image pixels per grid point.
 */
__device__ Point3D getImageCoord(int3 PointLoc, lr_basis basis, Point3D S, int scal){
	float3 Ploc = make_float3(PointLoc);
	Point3D Pim = (Ploc.x*basis.e1) + (Ploc.y*basis.e2) + (Ploc.z*basis.n) + S; // float3 + Point3D. The float3 will be converted to Point3D by trimming.

	// If Pim is outside
	Pim = Pim*scal;
	return Pim;
}

__host__ __device__ int3 getLrSize(int3 lrExtent){
	return lrExtent*2+(ONES3.getInt3());
}
