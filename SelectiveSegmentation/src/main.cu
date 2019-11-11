#include <iostream>
#include <string>
#include "macros.h"

#include <boost/chrono.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>

#include "cudatools/function.cuh"
#include "cudatools/deviceChunk.cuh"
#include "imagetools.cuh"
#include "common.cuh"
#include "cudatools/errorHandling.cuh"
#include "segmentation.cuh"
#include "SimpleConfig.cuh"

// external interface
#include "selective.h"

#define OUT_PATH_PROPERTY "data.out_path"

bool erro = false;

// input
int input_image_threshold;
int borderSize = 0;

SimpleConfig conf;

int initStrategy = 0;
int evolutionStrategy = 0;

AlgParams algParams;
GridParams gridParams;
Obj prior;

SegContext *segContext;

HostUByteChk *h_inputRawImage;
HostUByteChk *h_inputInitialMask;

EXPORT_SHARED p_int3 getPInt3(int3 a){
	p_int3 result;
	result.x = a.x;
	result.y = a.y;
	result.z = a.z;
	return result;
}

void segStep(){
    auto begin = boost::chrono::high_resolution_clock::now();

    gridParams.gridSize = segContext->resizeOptimalIfNeeded();
    segContext->iterate();

    auto end = boost::chrono::high_resolution_clock::now();
    auto dur = end - begin;
    auto ms = boost::chrono::duration_cast<boost::chrono::milliseconds>(dur).count();
}

void apply_settings(SimpleConfig& conf){

	// Input image
	borderSize = conf.getIProperty("img.pad");
	input_image_threshold = conf.getIProperty("img.thresh");
	initStrategy = conf.getIProperty("init.strategy");

	// Grid properties
	gridParams.gridRes = conf.getIProperty("grid.gridRes");

	// Rescale the grid in every step?
	evolutionStrategy = conf.getIProperty("ls.evolutionStrategy");

	// Algorithm weights for the selective model
	algParams.lambda=conf.getFProperty("eq.lambda");
	algParams.mu=conf.getFProperty("eq.mu");
	algParams.eta=conf.getFProperty("eq.eta");
	algParams.theta=conf.getFProperty("eq.theta");

	// Data term selection: edge or region based
	string data_term_val =conf.getSProperty("eq.data");
	if(data_term_val == DATA_TERM_LAPLACE_VAL){
		algParams.dataTerm = algParams.DATA_TERM_LAPLACE;
		BOOST_LOG_TRIVIAL(info) << "Data term: Laplace";
	}else if(data_term_val == DATA_TERM_LOCAL_INTEGRAL_VAL){
		algParams.dataTerm = algParams.DATA_TERM_LOCAL_INTEGRAL;
		BOOST_LOG_TRIVIAL(info) << "Data term: Local integral";
	}

	// Model selection: classic or selective
	string seg_model = conf.getProperty("eq.model");
	if(seg_model == MODEL_CLASSIC_VAL){
		algParams.model = algParams.MODEL_CLASSIC;
		BOOST_LOG_TRIVIAL(info) << "Segmentation model: Classic data(laplace|local integral) + smooth(Sum curvature|Euler elastica)";
	}else if(seg_model == MODEL_SELECTIVE_VAL){
		algParams.model = algParams.MODEL_SELECTIVE;
		BOOST_LOG_TRIVIAL(info) << "Segmentation model: Selective data(laplace|local integral) + smooth(Euler elastica) + volume(inflection|local minimum) + shape(plasma)";
	}

	// Region data term parameters
	algParams.regionExtent = make_int3(
			conf.getIProperty("eq.data.regionhw"),
			conf.getIProperty("eq.data.regionhh"),
			conf.getIProperty("eq.data.regionhd"));

	// Initialization with sphere parameters
	algParams.sphere_center = make_float3(
			conf.getFProperty("ls.init.sphere.center.x"),
			conf.getFProperty("ls.init.sphere.center.y"),
			conf.getFProperty("ls.init.sphere.center.z"));
	algParams.sphere_r = conf.getFProperty("ls.init.sphere.r");

	// Phase field parameters
	algParams.w=conf.getFProperty("phase.w");
	algParams.initRegCount=conf.getIProperty("phase.initRegCount");
	algParams.evolutionRegCount=conf.getIProperty("phase.evolutionRegCount");
	algParams.regNormTerm=conf.getFProperty("phase.regNormTerm");

	// Priors
	prior.vol=conf.getFProperty("pref.vol");
	prior.p=conf.getFProperty("pref.p");
}

EXPORT_SHARED int segmentation_app_headless_init(const void* image, const void* segmentation, int labelId, int pixelSize, p_int3 p_imageSize, SimpleConfig a_conf){
	try{
		conf = a_conf;
		conf.generateConfig(cout);

		int3 a_imageSize = make_int3(p_imageSize.x, p_imageSize.y, p_imageSize.z);
		int nPixelsImage = mul(a_imageSize);

		// Input image
		h_inputRawImage = new HostUByteChk(nPixelsImage);
		h_inputRawImage->copyFromN((uint8_t*)image, nPixelsImage);

		// Initial segmentation
		h_inputInitialMask = new HostUByteChk(nPixelsImage);
		h_inputInitialMask->fill(0);

		// Convert the mask given to short!
		uint16_t *shSeg = (uint16_t*) segmentation;
		for(int v = 0; v < nPixelsImage; v++){
			uint8_t val = (uint8_t) shSeg[v];
			if(val == uint8_t(labelId)){
				val = uint8_t(1);
			}else{
				val = uint8_t(0);
			}
			(*h_inputInitialMask)[v] = val;
		}

		// Update the global variables with the updated config
		apply_settings(conf);

		BOOST_LOG_TRIVIAL(info) << "Constructing image from a memory chunk.";
		unique_ptr<HostFloatChk> processedImg = UByteImagePreprocessor(h_inputRawImage, a_imageSize, input_image_threshold).getResult();

		BOOST_LOG_TRIVIAL(info) << "Creating segmentation context. " << endl;
		if(conf.isSetProperty("init.strategy")){
			initStrategy = conf.getIProperty("init.strategy");
		}
		segContext = new SegContext(*processedImg, a_imageSize, borderSize, algParams, gridParams, prior, initStrategy, evolutionStrategy, h_inputInitialMask);

		Size3D actualGridSize = segContext->getGridParams().gridSize;
		BOOST_LOG_TRIVIAL(info) << "Segmentation context created with grid size: " << actualGridSize;

		return 0;
	}catch(string& e){
		BOOST_LOG_TRIVIAL(fatal) << "Fatal error: " << e;
		return 1;
	}catch(int& e){
		BOOST_LOG_TRIVIAL(fatal) << "Fatal error: " << e;
		return 1;
	}catch(...){
		BOOST_LOG_TRIVIAL(fatal) << "Fatal error: unknown exception";
		return 1;
	}
}

EXPORT_SHARED p_ObjectStat segmentation_app_headless_step(SimpleConfig a_conf){
	p_ObjectStat objectStatReturn;

	try{
		algParams.updateFromConfig(a_conf);
		prior.updateFromConfig(a_conf);
		segContext->updateAlgParams(algParams);
		segContext->updatePrior(prior);

		gridParams.gridSize = segContext->getGridParams().gridSize;
		segStep();
		if(erro){
			BOOST_LOG_TRIVIAL(fatal) << "CUDA error in " <<  __FILE__ << ", exiting from the app! ";
		}
		gridParams.gridSize = segContext->getGridParams().gridSize;
		objectStatReturn.vol = segContext->getObjects()->vol;
		objectStatReturn.surf = segContext->getObjects()->surf;

		return objectStatReturn;
	}catch(string& e){
		BOOST_LOG_TRIVIAL(fatal) << "Fatal error: " << e;
	}catch(int e){
		BOOST_LOG_TRIVIAL(fatal) << "Fatal error: " << e;
	}catch(...){
		BOOST_LOG_TRIVIAL(fatal) << "Fatal error: unknown exception";
	}

	return objectStatReturn;

}

/**
 * A copy is made, and the caller is the new owner of the resource!
 */
EXPORT_SHARED float* segmentation_app_grab_level_set(p_int3& gsize, p_int3& trans){
	float *cop = new float[segContext->getLevelSetView()->getElements()];
	hchunk_float::uptr_t ls = segContext->getLevelSetView();
	ls->copyHostToHost(cop, ls->getPtr(), ls->getElements());
	gsize = getPInt3(gridParams.gridSize.geti3());
	trans = getPInt3(segContext->getGridToImageTranslation());
	return cop;
}

EXPORT_SHARED int segmentation_app_headless_cleanup(){
	BOOST_LOG_TRIVIAL(info) << "Cleaning up...";

	if(segContext != nullptr){
		delete segContext;
	}
	return 0;
}
