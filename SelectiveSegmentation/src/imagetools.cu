#include <iostream>
#include <string>
#include <sstream>
#include <math.h>

#include <thrust/reduce.h>

#include <boost/log/trivial.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "imagetools.cuh"

#include "cudatools/errorHandling.cuh"
#include "common.cuh"
#include "cudatools/deviceChunk.cuh"

#define EXT ".tif"

//using namespace std;

/*
 * Image processor: for 16/8-bit. Converts the raw image to a function for the algorithm.
 */

UshortImagePreprocessor::UshortImagePreprocessor(HostChunk<uint16_t>* aInputImage, Size3D aImageDims, float thresholdLevel){
	inputImage = aInputImage;
	imageDims = aImageDims;
	threshold = thresholdLevel;
}

unique_ptr<HostFloatChk> UshortImagePreprocessor::getResult(){
	auto result = HostFloatChk::make_uptr(imageDims.vol());
	DevFloatChk tmp(imageDims.vol());
	DeviceChunk<uint16_t> d_inputImage(*inputImage);
	launchConvertImage(tmp.getPtr(), d_inputImage.getPtr(), imageDims, threshold);

	float inputMaxIntensity = thrust::reduce(tmp.getTPtr(), tmp.getTPtr() + imageDims.vol(), -1, thrust::maximum<float>());
	launchScaleImage(tmp.getPtr(), inputMaxIntensity, imageDims);
	tmp.copyHost(*result);
	return result;
}

UByteImagePreprocessor::UByteImagePreprocessor(HostUByteChk* aInputImage, Size3D aImageDims, float thresholdLevel){
	inputImage = aInputImage;
	imageDims = aImageDims;
	threshold = thresholdLevel;
}

unique_ptr<HostFloatChk> UByteImagePreprocessor::getResult(){
	BOOST_LOG_TRIVIAL(info) << "getResult called. ImageDims: " <<  imageDims;
	auto result = HostFloatChk::make_uptr(imageDims.vol());
	DevFloatChk tmp(imageDims.vol());
	DeviceChunk<uint8_t> d_inputImage(*inputImage);
	launchConvertImage(tmp.getPtr(), d_inputImage.getPtr(), imageDims, threshold);

	float inputMaxIntensity = thrust::reduce(tmp.getTPtr(), tmp.getTPtr() + imageDims.vol(), -1, thrust::maximum<float>());
	launchScaleImage(tmp.getPtr(), inputMaxIntensity, imageDims);
	tmp.copyHost(*result);
	return result;
}

string get_stack_element_path(string path_stem, int element_id){
	stringstream stream;
	stream << path_stem << "/" << element_id << EXT;
	return stream.str();
}

/*
 * Read/write sliced tiffs with OpenCV.
 */

// Helpers:

std::string getImageType(int number)
{
    // find type
    int imgTypeInt = number%8;
    std::string imgTypeString;

    switch (imgTypeInt)
    {
        case 0:
            imgTypeString = "8U (8-bit unisgned)";
            break;
        case 1:
            imgTypeString = "8S (8-bit signed)";
            break;
        case 2:
            imgTypeString = "16U (16-bit unsigned)";
            break;
        case 3:
            imgTypeString = "16S (16-bit signed)";
            break;
        case 4:
            imgTypeString = "32S (32-bit signed)";
            break;
        case 5:
            imgTypeString = "32F (32-bit float)";
            break;
        case 6:
            imgTypeString = "64F (64-bit float)";
            break;
        default:
            break;
    }

    // find channel
    //int channel = (number/8) + 1;
    return imgTypeString;
}

Size3D getStackSize(vector<cv::Mat> stack){
	int w = 0;
	int h = 0;
	if(stack.size() > 0){
		w = stack[0].cols;
		h = stack[0].rows;
	}
	return Size3D(w, h, stack.size());
}

void raiseException(string filename){
	throw "Can't load the image \"" + filename + "\"";
}

void checkImreadError(cv::Mat& result, string filename){
	if(result.data == nullptr){
		raiseException(filename);
	}
}

void checkImreadMultiError(bool success, string filename){
	if(success != true){
		raiseException(filename);
	}
}

void checkImwriteError(bool success, string filename){
	if(success != true){
		throw "Can't write image to file \"" + filename + "\"";
	}
}

// Read the first stack element for metadata.

void CVTiffStackReader::getSESize(string stack_path_stem, Size3D& imageDims){
	string first_stack_element_path = get_stack_element_path(stack_path_stem, 1);

	//cv::Mat first_stack_element = cv::imread(first_stack_element_path, cv::CV_LOAD_IMAGE_ANYDEPTH);
	cv::Mat first_stack_element = cv::imread(first_stack_element_path, cv::IMREAD_ANYDEPTH);

	checkImreadError(first_stack_element, first_stack_element_path);

	if(first_stack_element.data == NULL){
		throw "Can't load the first stack element " + first_stack_element_path + "!";
	}
	imageDims.width = first_stack_element.cols;
	imageDims.height = first_stack_element.rows;
	BOOST_LOG_TRIVIAL(info) << "Checking the first stack element: dimX=" << imageDims.width << ", dimY=" << imageDims.height;
}

// Read 16-bit TIFF slice by slice.

unique_ptr<HostUShortChk> CVTiffStackReader::read(string filename_root, Size3D& imgDims, int count){
	string filename = get_stack_element_path(filename_root, 1);
	BOOST_LOG_TRIVIAL(info) << "Reading stack (single element TIFF, 16-bit): " << filename;

	getSESize(filename_root, imgDims);
	imgDims.depth = count;
	unique_ptr<HostUShortChk> image = HostUShortChk::make_uptr(imgDims.vol());

	for(int k = 0; k < imgDims.depth; k++){
		filename = get_stack_element_path(filename_root, k);

		string stack_element_path = get_stack_element_path(filename_root, k+1);

		cv::Mat stack_element = cv::imread(stack_element_path, CV_16UC1);
		checkImreadError(stack_element, stack_element_path);

		if(stack_element.data == NULL){
			throw "Can't load the stack element " + stack_element_path;
		}
		uint16_t* h_image = (*image).getPtr();
		memcpy(h_image + k*imgDims.width*imgDims.height, stack_element.data, imgDims.width*imgDims.height*2);
		BOOST_LOG_TRIVIAL(info) << "\tz=" << k << "," << filename << "(" << stack_element.cols << "," << stack_element.rows << ")";
	}
	BOOST_LOG_TRIVIAL(info) << "Image input done.";
	BOOST_LOG_TRIVIAL(info) << "Dimensions: dimX=" << imgDims.width << ", dimY=" << imgDims.height << ", dimZ=" << imgDims.depth;
	return image;

}

/*
 * Read/Write multiple page TIFF files using OpenCV (>=3.4).
 */

// Pads a stack.
vector<cv::Mat> CVMultiPageTiffIO::padStack(vector<cv::Mat> inputImage, int padSize){
	BOOST_LOG_TRIVIAL(info) << "Padding the input image with size: " << padSize;
	int resultWidth = inputImage[0].cols + 2*padSize;
	int resultHeight = inputImage[0].rows + 2*padSize;
	int resultDepth = inputImage.size() + 2*padSize;
	vector<cv::Mat> result;

	for(int z = 0; z < padSize; z++){
		result.push_back(cv::Mat::zeros(resultHeight, resultWidth, CV_8UC1));
	}

	for(int z = 0; z < inputImage.size(); z++){
		int srcSlice = z;
		//int dstSlice = z+padSize;
		cv::Mat dst(resultWidth, resultHeight, CV_8UC1);
		copyMakeBorder(
				inputImage[srcSlice],
				dst,
				padSize, padSize, padSize, padSize,
				cv::BORDER_CONSTANT, 0);
		result.push_back(dst);
	}

	for(int z = resultDepth-padSize; z < resultDepth; z++){
		result.push_back(cv::Mat::zeros(resultHeight, resultWidth, CV_8UC1));
	}

	return result;
}

vector<cv::Mat> CVMultiPageTiffIO::clipStack(vector<cv::Mat> inputImage, int clipSize){
	vector<cv::Mat> result;
	int z = clipSize;
	while(z < inputImage.size()-clipSize){
		cv::Mat src = inputImage[z];
		cv::Rect plane;
		plane.x = clipSize;
		plane.y = clipSize;
		plane.width = src.cols-2*clipSize;
		plane.height = src.rows-2*clipSize;
		result.push_back(src(plane));
		z++;
	}
	return result;
}

// Simply reads a stack.
vector<cv::Mat> CVMultiPageTiffIO::readStack(string filename){
	BOOST_LOG_TRIVIAL(info) << "Reading stack: " << filename << "...";
	vector<cv::Mat> stack;
	bool success = cv::imreadmulti(filename, stack, cv::IMREAD_UNCHANGED);
	if(stack.size() < 1){
		throw "The stack contains no slices, can not determine the pixel format!";
	}
	BOOST_LOG_TRIVIAL(info) << "Done. (" <<
			"channels=" << stack[0].channels() <<
			", pixel_type=" << getImageType(stack[0].depth()) <<
			", dimensions(cols,rows,n)=" <<
				stack[0].cols << "," <<
				stack[0].rows << "," <<
				stack.size() << ")";

	checkImreadMultiError(success, filename);
	return stack;
}

// Converts a stack to a flat 8-bit memory chunk.

hchunk<uint8_t>::uptr_t CVMultiPageTiffIO::getDeviceChunk8(vector<cv::Mat> stack, Size3D& imageDims){
	BOOST_LOG_TRIVIAL(info) << "Converting the image to device chunk (8 bits).";
	if(stack.size() < 1){
		throw "The stack contains no slices, can not determine the XY size!";
	}

	imageDims = getStackSize(stack);

	hchunk<uint8_t>::uptr_t image = hchunk<uint8_t>::make_uptr(imageDims.vol());
	uint8_t* h_image = image->getPtr();

	for(int k = 0; k < imageDims.depth; k++){
		int nPointsStack = imageDims.width*imageDims.height;
		memcpy(h_image + k*nPointsStack, stack[k].data, nPointsStack*sizeof(uint8_t));
	}
	return image;
}

// It could be done by using templates, but I do not want to spend more time to understand what the hell is going on behind the scenes in C++
// so I am duplicating the code.
hchunk<uint16_t>::uptr_t CVMultiPageTiffIO::getDeviceChunk16(vector<cv::Mat> stack, Size3D& imageDims){
	BOOST_LOG_TRIVIAL(info) << "Converting the image to device chunk (16 bits).";
	if(stack.size() < 1){
		throw "The stack contains no slices, can not determine the XY size!";
	}

	imageDims = getStackSize(stack);

	hchunk<uint16_t>::uptr_t image = hchunk<uint16_t>::make_uptr(imageDims.vol());
	uint16_t* h_image = image->getPtr();

	for(int k = 0; k < imageDims.depth; k++){
		int nPointsStack = imageDims.width*imageDims.height;
		memcpy(h_image + k*nPointsStack, stack[k].data, nPointsStack*sizeof(uint16_t));
	}
	return image;
}

// Reads an image and converts it to a 8-bit chunk.

hchunk_uint8::uptr_t CVMultiPageTiffIO::read8(string filename, Size3D& imgDims){
	vector<cv::Mat> stack = readStack(filename);
	auto image = getDeviceChunk8(stack, imgDims);
	return image;
}

// Read 16-bit TIFF volume.

unique_ptr<HostUShortChk> CVMultiPageTiffIO::read(string filename, Size3D& imgDims){
	vector<cv::Mat> stack_elements = readStack(filename);

	imgDims = getStackSize(stack_elements);

	unique_ptr<HostUShortChk> image = HostUShortChk::make_uptr(imgDims.vol());
	uint16_t* h_image = (*image).getPtr();

	for(int k = 0; k < imgDims.depth; k++){
		int nPointsStack = imgDims.width*imgDims.height;
		int nBytesPixel = 1;
		memcpy(h_image + k*nPointsStack, stack_elements[k].data, nPointsStack*nBytesPixel);
	}
	return image;
}

// Create a CV_8U Mat from a flat HostUByteChunk

vector<cv::Mat> CVMultiPageTiffIO::getStack8(HostUByteChk& data, Size3D imgDims){
	vector<cv::Mat> volume;

	for(int k = 0; k < imgDims.depth; k++){
		cv::Mat mat(imgDims.height, imgDims.width, CV_8U);
		for(int x = 0; x < imgDims.width; x++){
			for(int y = 0; y < imgDims.height; y++){
				int idx = x+y*imgDims.width+k*imgDims.width*imgDims.height;
				mat.at<uint8_t>(y, x) = data[idx];
			}
		}
		volume.push_back(mat);
	}
	return volume;
}

// Create a CV_16U Mat from a flat HostUShortChunk
vector<cv::Mat> CVMultiPageTiffIO::getStack16(hchunk_uint16& data, Size3D imgDims){
	vector<cv::Mat> volume;

	for(int k = 0; k < imgDims.depth; k++){
		cv::Mat mat(imgDims.height, imgDims.width, CV_16U);
		for(int x = 0; x < imgDims.width; x++){
			for(int y = 0; y < imgDims.height; y++){
				int idx = x+y*imgDims.width+k*imgDims.width*imgDims.height;
				mat.at<uint16_t>(y, x) = data[idx];
			}
		}
		volume.push_back(mat);
	}
	return volume;
}

// Writes a cv stack

void CVMultiPageTiffIO::writeStack(string filename, vector<cv::Mat> stack){
	BOOST_LOG_TRIVIAL(info) << "Writing stack (multipage TIFF): " << filename;
	bool succes = imwrite(filename, stack);
	checkImwriteError(succes, filename);
	BOOST_LOG_TRIVIAL(info) << "Image output done.";
}

void CVMultiPageTiffIO::write8(string filename, HostUByteChk& data, Size3D imgDims){
	vector<cv::Mat> stack = getStack8(data, imgDims);
	writeStack(filename, stack);
}

/**
 * Basic image processing tools.
 */

// Create a binary segmentation result from the actual state of the level set.

__global__ void segmentationFromLevelSet(float *d_level_set, uint8_t *binary_segmentation_result, Size3D gridDims){

	Point3D p = getThread3D();

	FltFunc3D levelSet(gridDims, d_level_set);
	Func3D<uint8_t> segmentationResult(gridDims, binary_segmentation_result);

	if(p >= 0 && p < (Point3D) gridDims){
		if(levelSet[p] < 0){
			segmentationResult[p] = INT_OBJ;
		}else{
			segmentationResult[p] = INT_BACK;
		}
	}
}

__device__ int3 convertCoords(int3 coord, int3 gridDims, int3 imDims) {
	return make_int3(coord.x*(imDims.x/gridDims.x), coord.y*(imDims.y/gridDims.y), coord.z*(imDims.z/gridDims.z));
}

// Initialise the level set by an image using thresholding.

__global__ void thresholdImage(float *d_data, float *d_image, float threshold, int3 gridDims, int3 imDims){

	// calculating 3D indices
	// (i, j, k) is the 3D index of the data array
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	if(0 <= i && i < gridDims.x && 0 <= j && j < gridDims.y && 0 <= k && k < gridDims.z){

		int gridIdx = i + j*gridDims.x + k*gridDims.x*gridDims.y;
		int3 imCoord = convertCoords(make_int3(i,j,k), gridDims, imDims);
		int imIdx = imCoord.x + imCoord.y*imDims.x + imCoord.z*imDims.x*imDims.y;

		if(d_image[imIdx] >= threshold){
			d_data[gridIdx] = -1;
		} else {
			d_data[gridIdx] = 1;
		}
	}
	return;
}

void launchThresholdImage(float *d_data, float *d_image, float threshold, int3 gridDims, int3 imDims){
	dim3 blockSize(8,8,8);
	dim3 gridSize((gridDims.x + 7)/8, (gridDims.y + 7)/8, (gridDims.z + 7)/8);
	thresholdImage<<<gridSize, blockSize>>>(d_data, d_image, threshold, gridDims, imDims);
	KERNEL_ERROR_CHECK("thresholdImage error");
}

// functions for converting images

__global__ void convertImage(float *d_image, unsigned short *d_intimage, int dimX, int dimY, int dimZ, int threshold){

	// calculating 3D indices
	// (i, j, k) is the 3D index of the data array
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	if(0 <= i && i < dimX && 0 <= j && j < dimY && 0 <= k && k < dimZ){
		int idx = LIN_IDX;
		if(d_intimage[idx] > threshold){
			d_image[idx] = ((float) d_intimage[idx]);
		}else{
			d_image[idx] = 0.0f;
		}
	}
	return;
}

__global__ void convertImage(float *d_image, uint8_t *d_intimage, int dimX, int dimY, int dimZ, int threshold){

	// calculating 3D indices
	// (i, j, k) is the 3D index of the data array
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	if(0 <= i && i < dimX && 0 <= j && j < dimY && 0 <= k && k < dimZ){
		int idx = LIN_IDX;
		if(d_intimage[idx] > threshold){
			d_image[idx] = ((float) d_intimage[idx]);
		}else{
			d_image[idx] = 0.0f;
		}
	}
	return;
}

void launchConvertImage(float *d_image, uint16_t *d_intimage, Size3D& imgDims, int threshold){
	int dimX = imgDims.width;
	int dimY = imgDims.height;
	int dimZ = imgDims.depth;

	dim3 blockSize(8,8,8);
	dim3 gridSize((dimX + 7)/8, (dimY + 7)/8, (dimZ + 7)/8);
	convertImage<<<gridSize, blockSize>>>(d_image, d_intimage, dimX, dimY, dimZ, threshold);
	KERNEL_ERROR_CHECK("convertImage error");
}

void launchConvertImage(float *d_image, uint8_t *d_intimage, Size3D& imgDims, int threshold){
	int dimX = imgDims.width;
	int dimY = imgDims.height;
	int dimZ = imgDims.depth;

	dim3 blockSize(8,8,8);
	dim3 gridSize((dimX + 7)/8, (dimY + 7)/8, (dimZ + 7)/8);
	BOOST_LOG_TRIVIAL(info) << "Size: " << dimX << " " << dimY << " " << dimZ;
	convertImage<<<gridSize, blockSize>>>(d_image, d_intimage, dimX, dimY, dimZ, threshold);
	KERNEL_ERROR_CHECK("convertImage error");
}

// functions for scaling images

__global__ void findMax(float *d_image, float *d_max, int dimX, int dimY, int dimZ){
	d_max[0] = d_image[0];
	for(int i = 1; i < dimX*dimY*dimZ; i++){
		if(d_image[i] > d_max[0]){
			d_max[0] = d_image[i];
		}
	}
}

void launchFindMax(float *d_image, float *d_max, int dimX, int dimY, int dimZ){
	findMax<<<1,1>>>(d_image, d_max, dimX, dimY, dimZ);
	KERNEL_ERROR_CHECK("findMax error");
}

__global__ void scaleImage(float *d_image, float scale, int dimX, int dimY, int dimZ){

	// calculating 3D indices
	// (i, j, k) is the 3D index of the data array
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;
	
	if(i < dimX && j < dimY && k < dimZ){
		d_image[i + j*dimX + k*dimX*dimY] = d_image[i + j*dimX + k*dimX*dimY]/scale;
	}
	return;
}

void launchScaleImage(float *d_image, float scale, Size3D dims){
	int dimX = dims.width;
	int dimY = dims.height;
	int dimZ = dims.depth;

	dim3 blockSize(8,8,8);
	dim3 gridSize((dimX + 7)/8, (dimY + 7)/8, (dimZ + 7)/8);
	scaleImage<<<gridSize, blockSize>>>(d_image, scale, dimX, dimY, dimZ);
	KERNEL_ERROR_CHECK("scaleImage error");
}

// functions for converting level set isosurfaces to images

__global__ void levelSetToImage(float *d_levelSet, int *d_image, int dimX, int dimY, int dimZ)
{
	// calculating 3D indices
	// (i, j, k) is the 3D index of the data array
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;
	int idx = i + dimX*j + dimX*dimY*k;

	if(i >= 0 && i < dimX && j >= 0 && j < dimY && k >= 0 && k < dimZ) {
		if(d_levelSet[idx] < 0) {
			d_image[idx] = 255;
		} else {
			d_image[idx] = 0;
		}
	}
}

void launchLevelSetToImage(float *d_levelSet, int *d_image, int dimX, int dimY, int dimZ)
{
	dim3 blockSize(8,8,8);
	dim3 gridSize((dimX + 7)/8, (dimY + 7)/8, (dimZ + 7)/8);
	levelSetToImage<<<gridSize, blockSize>>>(d_levelSet, d_image, dimX, dimY, dimZ);
	KERNEL_ERROR_CHECK("convertToImage error");

}

////////////////////////////////////////////////////////////////////////////////
// Initializing images and level set data from images
////////////////////////////////////////////////////////////////////////////////

// functions for initializing some built-in test images

__global__ void initializeImage(float *d_image, int dimX, int dimY, int dimZ){

	// calculating 3D indices
	// (i, j, k) is the 3D index of the data array
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;
	
	if(i < dimX && j < dimY && k < dimZ){
		// cylinder at center
		/*if((i-16)*(i-16) + (j-16)*(j-16) < 16){
			d_image[LIN_IDX] = 1;
		} else {
			d_image[LIN_IDX] = 0;
		}*/				
		/*
		// cube at center
		if(i > 10 && i < 21 && j > 10 && j < 21 && k > 10 && k < 21){
			d_image[LIN_IDX] = 1;
		} else {
			d_image[LIN_IDX] = 1;
		}*/
		
		// sphere at center
		/*if((i-16)*(i-16) + (j-16)*(j-16) + (k-16)*(k-16) < 49){
		} else {
			d_image[LIN_IDX] = 0;
		}*/
	}
	return;
}

void launchInitializeImage(float *d_image, int dimX, int dimY, int dimZ){
	dim3 blockSize(8,8,8);
	dim3 gridSize((dimX + 7)/8, (dimY + 7)/8, (dimZ + 7)/8);
	initializeImage<<<gridSize, blockSize>>>(d_image, dimX, dimY, dimZ);
	KERNEL_ERROR_CHECK("initializeImage error");
}

// functions for initializing built-in level sets

__global__ void initializeLevelSet(float *d_data, int dimX, int dimY, int dimZ){

	// calculating 3D indices
	// (i, j, k) is the 3D index of the data array
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	if(i < dimX && j < dimY && k < dimZ){

		// frame of the image 
		/*
		if(i == 0 || i == dimX-1 || j == 0 || j == dimY-1 || k == 0 || k == dimZ-1){
			d_data[LIN_IDX] = 5;
		} else {
			d_data[LIN_IDX] = -5;
		}
		*/

		// frame of the image signed distance function
		int coeff = _COEFF;
		d_data[LIN_IDX] = coeff*((2-min(min(min(i, dimX-1-i), min(j, dimY-1-j)), min(k, dimZ-1-k)))-0.5f);
		
		// double characteristic function of a cube
		/*if(i > 5 && i < dimX-6 && j > 5 && j < dimY-6 && k > 5 && k < dimZ-6) {
			d_data[LIN_IDX] = -1;
		} else {
			d_data[LIN_IDX] = 1;
		}*/
		
		// double characteristic function of a sphere
		/*if((i-15.5)*(i-15.5) + (j-15.5)*(j-15.5) + (k-15.5)*(k-15.5) < 400) {
			d_data[LIN_IDX] = -1;
		} else {
			d_data[LIN_IDX] = 1;
		}*/
		
		// nonconvex blob
		/*d_data[LIN_IDX] =  min(min(( (i-16)*(i-16)/3 + (j-16)*(j-16)/4 + (k-16)*(k-16) )*0.01 - 0.15,
											    ( (i-20)*(i-20)   + (j-20)*(j-20)   + (k-20)*(k-20) )*0.01 - 0.8),
												( (i-10)*(i-10)   + (j-17)*(j-17)   + (k-8)*(k-8) )*0.01 - 0.45);*/
										
		// cylinder at center
		/*if((i-16)*(i-16) + (j-16)*(j-16) < 100 && k > 0 && k < dimZ-1){
			d_data[LIN_IDX] = -1;
		} else {
			d_data[LIN_IDX] = 1;
		}*/
		
		// ellipsoid at the center
		//d_data[LIN_IDX] = ((i-15.5)*(i-15.5)/(200) + (j-15.5)*(j-15.5)/(200) + (k-15.5)*(k-15.5)/(100) - 1);
		//d_data[LIN_IDX] = ((i-31.5)*(i-31.5)/(15*15) + (j-31.5)*(j-31.5)/(25*25) + (k-31.5)*(k-31.5)/(10*10) - 1);
		//d_data[LIN_IDX] = ((i-31.5)*(i-31.5)/(17*17) + (j-31.5)*(j-31.5)/(13*13) + (k-31.5)*(k-31.5)/(15*15) - 1);
	}
	return;
}

// Initialises a phase field
__global__ void initPhaseFieldSphere(float *field, Size3D a_size, float3 center, float r){
	func3_f f_field(a_size, field);
	int3 p = getThread3D().getInt3();
	int3 size = a_size.geti3();

	float phaseMinMax = 1.0f;

	if(p < size){
		float3 Pf = make_float3(p);
		float3 centermP = center-Pf;
		bool insideContour = norm3df(centermP.x, centermP.y, centermP.z) < r;

		if(insideContour){
			f_field[p] = phaseMinMax;
		}else{
			f_field[p] = -phaseMinMax;
		}
	}
	return;
}

__global__ void initPhaseFieldCube(float *field, Size3D a_size, float dst){
	func3_f f_field(a_size, field);
	int3 p = getThread3D().getInt3();
	int3 size = a_size.geti3();

	float phaseMinMax = 1.0f;

	if(p < size){
		float3 Pf = make_float3(p);
		float3 sizeF = make_float3(a_size.geti3());
		bool insideContour = Pf < (sizeF-make_float3(dst)) && Pf > make_float3(dst);

		if(insideContour){
			f_field[p] = phaseMinMax;
		}else{
			f_field[p] = -phaseMinMax;
		}
	}
}

void launchInitPhaseFieldMaxSphere(float* field, GridParams gProps, AlgParams algParams, int borderSize){
	float maxR = Point3D(gProps.gridSize).min()/3.0f - borderSize;
	float3 center = make_float3((gProps.gridSize.geti3()))/2.0f;
	GpuConf3D conf(gProps.gridSize, SQ_TPB_3D);
	initPhaseFieldSphere<<<conf.grid(), conf.block()>>>(field, gProps.gridSize, center, maxR);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

void launchInitPhaseFieldSphere(float* field, GridParams gProps, AlgParams algParams, float3 center, float r){
	GpuConf3D conf(gProps.gridSize, SQ_TPB_3D);
	initPhaseFieldSphere<<<conf.grid(), conf.block()>>>(field, gProps.gridSize, center, r);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

void launchInitPhaseFieldCube(float* field, GridParams gProps, AlgParams algParams, int dstFromBorder){
	GpuConf3D conf(gProps.gridSize, SQ_TPB_3D);
	initPhaseFieldCube<<<conf.grid(), conf.block()>>>(field, gProps.gridSize, dstFromBorder);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}


void launchInitializeLevelSet(float *d_data, GridParams gProps){
	int dimX = gProps.gridSize.width;
	int dimY = gProps.gridSize.height;
	int dimZ = gProps.gridSize.depth;

	dim3 blockSize(8,8,8);
	dim3 gridSize((dimX + 7)/8, (dimY + 7)/8, (dimZ + 7)/8);
	initializeLevelSet<<<gridSize, blockSize>>>(d_data, dimX, dimY, dimZ);
	KERNEL_ERROR_CHECK("initializeLevelSet error");
}

__global__ void launchInitPhaseFieldMaskKernel(float* field, uint8_t* mask, int3 size){
	int3 p = getThread3D().getInt3();
	func3_f f_field(size, field);
	func3<uint8_t> f_mask(size, mask);
	if(p < size){
		if(f_mask[p] == 1){
			f_field[p] = 1.0f;
		}else{
			f_field[p] = -1.0f;
		}
	}
}

void launchInitPhaseFieldMask(float* field, uint8_t* mask, GridParams gProps, AlgParams algParams){
	GpuConf3D conf(gProps.gridSize, SQ_TPB_3D);
	launchInitPhaseFieldMaskKernel<<<conf.grid(), conf.block()>>>(field, mask, gProps.gridSize.geti3());
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}
