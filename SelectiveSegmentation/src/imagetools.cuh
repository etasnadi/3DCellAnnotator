#ifndef IMAGETOOLS_CUH
#define IMAGETOOLS_CUH

#include <iostream>
#include <string>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cudatools/deviceChunk.cuh"
#include "common.cuh"

using namespace std;

class StackReader {
public:
	unique_ptr<HostUShortChk> read(string filename_root, Size3D& imgDims);
	unique_ptr<HostUByteChk> read8(string filename_root, Size3D& imgDims);
};

class CVTiffStackReader : StackReader {
private:
	void getSESize(string stack_path_stem, Size3D& imageDims);
public:
	unique_ptr<HostUShortChk> read(string filename_root, Size3D& imgDims, int count);
};

class CVMultiPageTiffIO : StackReader {
public:
	vector<cv::Mat> padStack(vector<cv::Mat> inputImage, int padSize);
	vector<cv::Mat> clipStack(vector<cv::Mat> inputImage, int clipSize);

	vector<cv::Mat> readStack(string filename);
	hchunk_uint8::uptr_t getDeviceChunk8(vector<cv::Mat> stack, Size3D& imageDims);
	hchunk<uint16_t>::uptr_t getDeviceChunk16(vector<cv::Mat> stack, Size3D& imageDims);

	unique_ptr<HostUShortChk> read(string filename_root, Size3D& imgDims);
	unique_ptr<HostUByteChk> read8(string filename_root, Size3D& imgDims);

	vector<cv::Mat> getStack8(HostUByteChk& data, Size3D imageDims);
	vector<cv::Mat> getStack16(hchunk_uint16& data, Size3D imgDims);

	void write8(string file_name, HostUByteChk& data, Size3D imgDims);
	void writeStack(string filenane, vector<cv::Mat> stack);
};

////////////////////////////////////////////////////////////////////////////////
// Image tools
////////////////////////////////////////////////////////////////////////////////

__global__ void segmentationFromLevelSet(float *d_level_set, uint8_t *binary_segmentation, Size3D gridDims);

// functions for initializing level set by thresholding images
__device__ int3 convertCoords(int3 coord, int3 gridDims, int3 imDims);
__global__ void thresholdImage(float *d_data, float *d_image, float threshold, int3 gridDims, int3 imDims);
void launchThresholdImage(float *d_data, float *d_image, float threshold, int3 gridDims, int3 imDims);

// functions for converting images
__global__ void convertImage(float *d_image, unsigned short *d_intimage, int dimX, int dimY, int dimZ, int threshold);
__global__ void convertImage(float *d_image, uint8_t *d_intimage, int dimX, int dimY, int dimZ, int threshold);
void launchConvertImage(float *d_image, uint16_t *d_intimage, Size3D& imgDims, int threshold);
void launchConvertImage(float *d_image, uint8_t *d_intimage, Size3D& imgDims, int threshold);

// functions for scaling images
__global__ void findMax(float *d_image, float *d_max, int dimX, int dimY, int dimZ);
void launchFindMax(float *d_image, float *d_max, int dimX, int dimY, int dimZ);
__global__  void scaleImage(float *d_image, float scale, int dimX, int dimY, int dimZ);
void launchScaleImage(float *d_image, float scale, Size3D dims);


////////////////////////////////////////////////////////////////////////////////
// Initializing images and level set data from images
////////////////////////////////////////////////////////////////////////////////

// functions for initializing some built-in test images
__global__  void initializeImage(float *d_image, int dimX, int dimY, int dimZ);
void launchInitializeImage(float *d_image, int dimX, int dimY, int dimZ);

// functions for initializing built-in level sets
__global__  void initializeLevelSet(float *d_data, int dimX, int dimY, int dimZ);
void launchInitializeLevelSet(float *d_data, GridParams gProps);

void launchInitPhaseFieldMaxSphere(float* field, GridParams gProps, AlgParams algParams, int borderSize);
void launchInitPhaseFieldSphere(float* field, GridParams gProps, AlgParams algParams, float3 center, float r);
void launchInitPhaseFieldCube(float* field, GridParams gProps, AlgParams algParams, int dstFromBorder);
void launchInitPhaseFieldMask(float* field, uint8_t* mask, GridParams gProps, AlgParams algParams);

class ImagePreprocessorBase {
public:
	unique_ptr<HostFloatChk> getResult();
};

class UshortImagePreprocessor : ImagePreprocessorBase {
private:
	HostChunk<uint16_t>* inputImage;
	Size3D imageDims;
	float threshold;
public:
	UshortImagePreprocessor(HostChunk<uint16_t>* aInputImage, Size3D aImageDims, float thresholdLevel);
	unique_ptr<HostFloatChk> getResult();
};

class UByteImagePreprocessor : ImagePreprocessorBase {
private:
	HostUByteChk* inputImage;
	Size3D imageDims;
	float threshold;
public:
	UByteImagePreprocessor(HostUByteChk* aInputImage, Size3D aImageDims, float thresholdLevel);
	unique_ptr<HostFloatChk> getResult();
};

#endif
