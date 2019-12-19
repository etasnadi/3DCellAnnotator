#include "segmentation.cuh"

#include "marchingCubes.cuh"
#include "cudatools/deviceChunk.cuh"
#include "evolve.cuh"
#include "imagetools.cuh"
#include "common.cuh"
#include "cudatools/errorHandling.cuh"
#include "surfaceAndVolume.cuh"
#include "buildGraph.cuh"
#include "ccalg/computeComponents.cuh"
#include "phaseField.cuh"
#include "narrowBand.cuh"

#include <thrust/reduce.h>
#include <boost/log/trivial.hpp>

using namespace thrust;

#define _K 3

void SegContext::initImageStructures(int imgPts){
	workImage = DevFloatChk::make_uptr(imgPts);
	dataTerm = DevFloatChk::make_uptr(imgPts);			// Data term for the laplacian...
	std::cout << "Allocating memory for the 1st and 2nd image derivatives... " << imgPts << std::endl;
	if(algParams.dataTerm == algParams.DATA_TERM_LOCAL_INTEGRAL){
		imgDer = DeviceChunk<FltDer3D>::make_uptr(imgPts);	// Image derivatives for the local region
	}
}

void SegContext::initLevelSetStructures(int gridPts){
	ls1 = DevFloatChk::make_uptr(gridPts);
	ls2 = DevFloatChk::make_uptr(gridPts);
}

void SegContext::initGridAuxStructures(int gridPts){
	std::cout << ":Enter Initialize grid aux structures. N grid points: " << gridPts << std::endl;
	closestObj = DevIntChk::make_uptr(gridPts);
	auxField = DevFloatChk::make_uptr(gridPts);
	normals = dchunk<float3>::make_uptr(gridPts);
	K = dchunk<float>::make_uptr(gridPts);
	std::cout << ":Leave Initialize grid aux structures" << std::endl;
}

void SegContext::updateGridStructures(int gridPts){
	closestObj = std::move(DevIntChk::make_uptr(gridPts));
	auxField = std::move(DevFloatChk::make_uptr(gridPts));
	normals = std::move(dchunk<float3>::make_uptr(gridPts));
	K = std::move(dchunk<float>::make_uptr(gridPts));

	ls1 = std::move(DevFloatChk::make_uptr(gridPts));
	ls2 = std::move(DevFloatChk::make_uptr(gridPts));
}

__global__ void copyTransKernel(func3_f f_dst, func3_f f_src, int3 trans){
	int3 p = getThread3D().getInt3();

	int3 dstSize = f_dst.getSize().geti3();
	int3 srcSize = f_src.getSize().geti3();

	if(p < dstSize && p-trans < srcSize && p-trans > IZEROS3){
		f_dst[p] = f_src[p-trans];
	}
}

std::pair<int, int> resize1D(int objMin, int objMax, int size){
	int k = _K;

	int newSize = size;
	int translation = 0;

	if((objMin < k || objMax > size-1-k) || (objMin>3*k || objMax < size-1-3*k)){
		newSize = objMax-objMin+4*k;
		translation = 2*k-objMin;
	}

	if(newSize < size && newSize < 6*k){
		newSize = size;
		translation = 0;
	}

	return std::pair<int, int>(newSize, translation);
}

std::pair<int3, int3> getResizeParameters(std::pair<int3, int3> objExtrema, int3 actSize){
	int3 objMin = objExtrema.first;
	int3 objMax = objExtrema.second;

	auto x = resize1D(objMin.x, objMax.x, actSize.x);
	auto y = resize1D(objMin.y, objMax.y, actSize.y);
	auto z = resize1D(objMin.z, objMax.z, actSize.z);

	int3 resize = make_int3(x.first, y.first, z.first);
	int3 translation = make_int3(x.second, y.second, z.second);

	return std::pair<int3, int3>(resize, translation);
}

Size3D SegContext::resizeOptimalIfNeeded(bool initialResize, int3 a_mins, int3 a_maxs){
	int3 mins;
	int3 maxs;

	if(initialResize){
		BOOST_LOG_TRIVIAL(info) << "Using the resize parameters given in the arguments...";
		mins = a_mins;
		maxs = a_maxs;
		BOOST_LOG_TRIVIAL(info) << "Min: " << mins <<  " max: " << maxs;
	}else{
		BOOST_LOG_TRIVIAL(info) << "Computing the resize parameters...";
		std::pair<int3, int3> ex = getFieldExtrema(*ls1, gridParams);
		mins = ex.first;
		maxs = ex.second;
	}

	BOOST_LOG_TRIVIAL(info) << "Actual grid size: " << gridParams.gridSize << ", content extrema: " << mins << "; " << maxs;

	auto params = getResizeParameters(std::pair<int3, int3>(mins, maxs), gridParams.gridSize.geti3());

	int3 scaleTransform = params.first; // Resize transform
	int3 tTrans = params.second; // Translate transform

	Size3D newSize = gridParams.gridSize;

	if(evolutionStrategy == 1){
		int3 srcSize = gridParams.gridSize.geti3();
		if(scaleTransform.x != srcSize.x || scaleTransform.y != srcSize.y || scaleTransform.z != srcSize.z){ // At least one of them is not zero, therefore we have to apply the transformation.
			newSize = Size3D(scaleTransform);
			auto ud_dst = dchunk_float::make_uptr(mul(scaleTransform));

			GpuConf conf = WorkManager(scaleTransform, 4).conf();
			func3_f fd_src = ls1->funcView(Size3D(srcSize));
			func3_f fd_dst = ud_dst->funcView(scaleTransform);
			ud_dst->fill(1.0f);
			copyTransKernel<<<conf.grid(), conf.block()>>>(fd_dst, fd_src, tTrans);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());

			gridParams.gridSize = newSize;
			gridToImageTranslation = gridToImageTranslation + tTrans;
			updateGridStructures(mul(scaleTransform));
			ud_dst->copy(*ls1);
			ud_dst->copy(*ls2);
		}
	}
	return newSize;
}

unique_ptr<HostUByteChk> SegContext::getBinarySegmentation(){
	auto segmentation_device = DevUByteChk::make_uptr(gridParams.gridSize.vol());
	GpuConf3D conf(gridParams.gridSize, 4);
	segmentationFromLevelSet<<<conf.grid(), conf.block()>>>(
			this->getLevelSet().getPtr(),
			segmentation_device->getPtr(), gridParams.gridSize);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	unique_ptr<HostUByteChk> segmentation_host = HostUByteChk::make_uptr(gridParams.gridSize.vol());
	segmentation_device->copyHost(*segmentation_host);
	return segmentation_host;
}

void negateField(DevFloatChk& ls1, DevFloatChk& ls2, int nElems){
	thrust::transform(ls1.getTPtr(), ls1.getTPtr()+nElems, ls1.getTPtr(), thrust::negate<float>());
	thrust::transform(ls2.getTPtr(), ls2.getTPtr()+nElems, ls2.getTPtr(), thrust::negate<float>());
}

void SegContext::initLevelSet(){
	BOOST_LOG_TRIVIAL(info) << "Phase field init (max sphere)";

	launchInitPhaseFieldMaxSphere(ls1->getPtr(), gridParams, algParams, 0);
	launchInitPhaseFieldMaxSphere(ls2->getPtr(), gridParams, algParams, 0);

	negateField(*ls1, *ls2, gridParams.gridSize.vol());
}

void SegContext::initLevelSetWithMask(hchunk<uint8_t>& mask){
	BOOST_LOG_TRIVIAL(info) << "Phase field init (mask)";
	dchunk<uint8_t> d_mask(mask.getElements());
	d_mask.copyDevice(mask.getPtr());

	launchInitPhaseFieldMask(ls1->getPtr(), d_mask.getPtr(), gridParams, algParams);
	launchInitPhaseFieldMask(ls2->getPtr(), d_mask.getPtr(), gridParams, algParams);

	negateField(*ls1, *ls2, gridParams.gridSize.vol());
}

void SegContext::initLevelSetWithEnclosingSphere(hchunk<uint8_t>& mask){
	BOOST_LOG_TRIVIAL(info) << "Phae field init (encolsing sphere)";

	dchunk<uint8_t> d_mask(mask.getElements());
	d_mask.copyDevice(mask.getPtr());

	launchInitPhaseFieldMask(ls1->getPtr(), d_mask.getPtr(), gridParams, algParams);
	launchInitPhaseFieldMask(ls2->getPtr(), d_mask.getPtr(), gridParams, algParams);

	std::pair<int3, int3> exs = getFieldExtrema(*ls1, gridParams);
	cout << "Initializing level set using the extrema: " << exs.first << " - " << exs.second << endl;

	int3 mins = exs.first;
	int3 maxs = exs.second;

	int3 sums = mins+maxs;

	float3 center = make_float3(float(sums.x/2), float(sums.y/2), float(sums.z/2));
	int3 diffs = maxs-mins;
	float r = (diffs.x+diffs.y+diffs.z)/6.0f;

	launchInitPhaseFieldSphere(ls1->getPtr(), gridParams, algParams, center, r);
	launchInitPhaseFieldSphere(ls2->getPtr(), gridParams, algParams, center, r);

	negateField(*ls1, *ls2, gridParams.gridSize.vol());
}


void SegContext::computeData(){
	computeDataTerm(dataTerm->getPtr(), workImage->getPtr(), imageDims);
	if(algParams.dataTerm == algParams.DATA_TERM_LOCAL_INTEGRAL){
		computeImgDerivatives(imgDer->getPtr(), workImage->getPtr(), imageDims);
	}
}

void SegContext::regularise(){
	BOOST_LOG_TRIVIAL(info) << "Phase field regularisation";
	negateField(*ls1, *ls2, gridParams.gridSize.vol());

	for(int i = 0; i < algParams.evolutionRegCount; i++){
		regularisePhaseField(ls1->getPtr(), gridParams, algParams);
		regularisePhaseField(ls2->getPtr(), gridParams, algParams);
	}

	negateField(*ls1, *ls2, gridParams.gridSize.vol());
}

void SegContext::evolve(){
	// Everything is computed based on the currently active level set!
	// Therefore we should visualize ls1 (the input level set), the K (computed from the input level set)
	BOOST_LOG_TRIVIAL(info) << "Iter id (n of iters performed): " << this->iterations;
	fillNeighborsArray(*ls1, *closestObj, gridParams.gridSize);
	ccGraph = buildGraph(*ls1, *closestObj, gridParams.gridSize);

	ccResult = computeCC(ccGraph, gridParams.gridSize.vol());
	int nComps = makeDense(ccResult, gridParams.gridSize.vol());
	graphComputed = true;

	findClosestObjectNB(closestObj->getPtr(), ccResult->getPtr(), 4, gridParams.gridSize);

	*objectStats = computeStats(ls1, ccResult, nComps, gridParams);
	printStats(*objectStats, nComps, std::cout);

	// Input: ls1, output: ls2
	BOOST_LOG_TRIVIAL(info) << "Grid to image translation: " << gridToImageTranslation;

	FltDer3D *imageDerivatives = nullptr;
	if(algParams.dataTerm == algParams.DATA_TERM_LOCAL_INTEGRAL){
		imageDerivatives = imgDer->getPtr();
	}

	launchEvolve(
		*ls2, *ls1, *workImage, dataTerm->getPtr(), imageDerivatives, gridToImageTranslation,
		*normals, *K,
		closestObj->getPtr(),
		*objectStats,
		gridParams, imageDims, algParams, iterations, preferredObject);


	// Now the input is in the ls2 and the output is in the ls1
	ls1.swap(ls2);
	float delta = objectStats->vol;
	BOOST_LOG_TRIVIAL(info) << "Delta: " << delta;
	nComps_sh = nComps; // commit nComps
	iterations++;

}

SegContext::SegContext(HostFloatChk& image, Size3D aImageDims, int aBorderSize, AlgParams aAlgParams, GridParams& aGridProps, Obj aPrefObj, int3 ex_min, int3 ex_max, int strategy, int a_evolutionStrategy, HostUByteChk* mask){
	// Init GPU constant tables
	Cubes::initGpuTables();

	borderSize = aBorderSize;
	algParams = aAlgParams;
	preferredObject = aPrefObj;
	imageDims = aImageDims;

	aGridProps.gridSize = aImageDims / aGridProps.gridRes;
	gridParams = aGridProps;


	initLevelSetStructures(gridParams.gridSize.vol());
	/**
	 * Level set initialization strategy:
	 * 0 -> using the provided mask
	 * 1 -> using a minimal enclosing sphere around the provided mask
	 * 2 -> using an object with the parameters given in the alg params.
	 */


	BOOST_LOG_TRIVIAL(info) << "Init strategy: " << strategy;
	if(strategy == 0){
		initLevelSetWithMask(*mask);
	}else if(strategy == 1){
		initLevelSetWithEnclosingSphere(*mask);
	}else{
		initLevelSet();
	}

	BOOST_LOG_TRIVIAL(info) << "Level set evolution strategy: " << a_evolutionStrategy;
	evolutionStrategy = a_evolutionStrategy;

	gridToImageTranslation = make_int3(0, 0, 0);
	resizeOptimalIfNeeded(true, ex_min, ex_max);
	initGridAuxStructures(gridParams.gridSize.vol());


	initImageStructures(imageDims.vol());
	workImage->copyDevice(image);

	computeData();

	objectStats = std::make_shared<CurveProps>();
}

void SegContext::iterate(){
	regularise();
	evolve();
}

// The returned data in the DevFloatChk will be invalidated upon the next iterate call!
DevFloatChk& SegContext::getLevelSet(){
	return *ls1;
}

unique_ptr<HostFloatChk> SegContext::getLevelSetView(){
	auto res = HostFloatChk::make_uptr(*ls1);
	return res;
}

unique_ptr<HostIntChk> SegContext::getObjectIds(){
	return HostIntChk::make_uptr(*closestObj);
}

DevFloatChk& SegContext::getImage(){
	return *workImage;
}

int SegContext::getIterations(){
	return iterations;
}

void SegContext::updateAlgParams(AlgParams aAlgParams){
	algParams = aAlgParams;
}

GridParams SegContext::getGridParams(){
	return gridParams;
}

void SegContext::updatePrior(Obj prior){
	preferredObject = prior;
}

int SegContext::getObjSummary(){
	return nComps_sh;
}

unique_ptr<HostIntPairChk> SegContext::getCCgraph(){
	if(!graphComputed){
		return HostIntPairChk::make_uptr(0);
	}
	return HostIntPairChk::make_uptr(*ccGraph);
}

unique_ptr<HostIntChk> SegContext::getCCResult(){
	if(!graphComputed){
		return HostIntChk::make_uptr(0);
	}
	return HostIntChk::make_uptr(*ccResult);
}

int3 SegContext::getGridToImageTranslation(){
	return gridToImageTranslation;
}

void SegContext::loadLevelSet(HostFloatChk& aLevelSet){
	ls1->copyDevice(aLevelSet);
}

shared_ptr<CurveProps> SegContext::getObjects(){
	return this->objectStats;
}
