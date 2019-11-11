#include <iostream>

#include "imagetools.cuh"
#include "cudatools/types.cuh"
#include "phaseField.cuh"
#include "surfaceAndVolume.cuh"
#include "marchingCubes.cuh"

#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <thrust/transform.h>
#include <thrust/reduce.h>

using namespace std;
using namespace thrust;

template<typename T>
struct phase_threshold{
	T lev;
	T minmax;

	phase_threshold(T a_lev, T a_minmax){
		lev = a_lev;
		minmax = a_minmax;
	}
  /*! \typedef argument_type
   *  \brief The type of the function object's argument.
   */
  typedef T argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef T result_type;

  /*! Function call operator. The return value is <tt>-x</tt>.
   */
  __host__ __device__ T operator()(const T &x) const {
	  if(x > lev){
		  return minmax;
	  }else{
		  return -minmax;
	  }
  }
}; // end negate

void getStats(dchunk_float& d_img, Size3D imageDims){
	float isoLevel = 0.5;
	float phaseMinmax = 1.0f;
	float nRegCycles = 20;

	AlgParams aParams;
	aParams.w = 4.0f;
	aParams.regNormTerm = 0.01f;

	GridParams gProps;
	gProps.gridSize = imageDims;
	gProps.gridRes = 1;

	// Init phase field from image
	thrust::transform(
			d_img.getTPtr(),
			d_img.getTPtr()+imageDims.vol(),
			d_img.getTPtr(), phase_threshold<float>(isoLevel, phaseMinmax)); // in-place transformation

	// Regluarise it!
	for(int i = 0; i < nRegCycles; i++){
		regularisePhaseField(d_img.getPtr(), gProps, aParams);
	}

	thrust::transform(d_img.getTPtr(), d_img.getTPtr()+gProps.gridSize.vol(), d_img.getTPtr(), thrust::negate<float>()); // in-place transformation

	DevFloatChk aux(gProps.gridSize.vol());

	Cubes::initGpuTables();
	float _surf = getSurface_global(d_img, aux, gProps.gridRes, gProps.gridSize);
	float _vol = getVolume_global(d_img, aux, gProps.gridRes, gProps.gridSize);

	BOOST_LOG_TRIVIAL(info) << "Volume: " << _vol << ", surface: " << _surf << ", plasma: " << pow(_surf, float(3.0f/2.0f))/_vol;

}

void getStatsGroundTruth(string& gt_path){
	Size3D imageDims;

	int borderSize = 0;
	int input_image_threshold = 0;

	auto img = CVMultiPageTiffIO().padStack(CVMultiPageTiffIO().readStack(gt_path), borderSize);
	unique_ptr<HostUByteChk> rawImage8 = CVMultiPageTiffIO().getDeviceChunk8(img, imageDims);
	auto processedImg = UByteImagePreprocessor(rawImage8.get(), imageDims, input_image_threshold).getResult();

	dchunk_float d_img = dchunk_float(*processedImg);
	getStats(d_img, imageDims);

}

int main(){
	BOOST_LOG_TRIVIAL(info) << "Contour measurement app.";
	string imgPath = "/home/ervin/docs/3D-images/FilippoProject/ManualSegmentations/PiccininiFilippo/tiff8/2012-12-21-crwn12-008_GC1.tif";
	getStatsGroundTruth(imgPath);

	imgPath = "/home/ervin/docs/3D-images/FilippoProject/ManualSegmentations/TothTimea/tiff8/2012-12-21-crwn12-008_GC1.tif";
	getStatsGroundTruth(imgPath);

	//imgPath = "/home/ervin/devel/3d-segmentation-build/Tue_04-Dec-2018_18.57.08/700_2012-12-21-crwn12-008_GC1.tif";
	//imgPath = "/home/ervin/devel/3d-segmentation-build/Tue_04-Dec-2018_18.57.08/700_2012-12-21-crwn12-008_GC1.tif";

	getStatsGroundTruth(imgPath);
	//launchInitPhaseField(d_img.getPtr(), gProps, aParams, 0);
	//initPhaseFieldFromImage(d_img, d_img, gProps.gridSize, isoLevel);
}
