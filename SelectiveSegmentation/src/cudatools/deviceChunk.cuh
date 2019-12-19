#ifndef _DEVICE_CHUNK_CUH
#define _DEVICE_CHUNK_CUH 1

#include <stdio.h>
#include <memory>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include "cudaErr.cuh"
#include "types.cuh"
#include "function.cuh"
#include "cuda.h"
#include <iostream>

//#include "cudacomp.cuh"
//#include "cuda.h"

extern size_t us;
extern size_t h_us;

// Detailed log of the device chunk
#ifndef _DC_LOG
#define _DC_LOG 1
#endif

#ifndef _HC_LOG
#define _HC_LOG 0
#endif

// Only print the memory allocations and releases accounted by the device chunk
#ifndef _DC_OP_LOG
#define _DC_OP_LOG 1
#endif

#ifndef _HC_OP_LOG
#define _HC_OP_LOG 0
#endif
// Request the memory statistics from the cuda runtime before and after every memory allocation/release
#ifndef _DC_OP_LOG_CUDA
#define _DC_OP_LOG_CUDA 1
#endif

#if _DC_OP_LOG_CUDA == 1
#include "cuda_runtime.h"
#endif

template<typename T>
class Chunk;

template<typename T>
class DeviceChunk;

template<typename T>
class DeviceVal;

template<typename T>
class HostChunk;

template<typename T>
class Chunk {
protected:
	T* pointer = nullptr;
	int elements = 0;

	__device__ __host__ void rangeCheck(int index);
public:
	Chunk() : elements(0),pointer (nullptr){}
	Chunk(int pElements) : elements(pElements),pointer(nullptr){}
	Chunk(int pElements,T* pPtr) : elements(pElements),pointer(pPtr){}
	Chunk(Chunk& other, bool unused) : elements(other.elements), pointer(other.pointer){};
	//Chunk(Chunk& other) = delete;
	__device__ __host__ T* getPtr();
	__device__ __host__ int getElements();

	Func3D<T> funcView(Size3D dims);
	Func2D<T> funcView(Size2D dims);
	Func1D<T> funcView(int nelems);
	Func1D<T> funcView();

};

// Low level memory functionalities

template<typename T>
class MemoryOperations {
public:

	static float getMegs(size_t bytes){
		return float(bytes)/float(1024*1024);
	}

    static void copyHostToHost(T* dst, T* src, size_t nElems) {
    	memcpy(dst, src, nElems*sizeof(T));
    }

    static void copyDeviceToDevice(T* dst, T* src, size_t nElems) {
		gpuErrchk(cudaMemcpy(dst, src, nElems*sizeof(T), cudaMemcpyDeviceToDevice));
	}

    static void copyHostToDevice(T* device_dst, T* host_src, size_t nElems) {
    	gpuErrchk(cudaMemcpy(device_dst, host_src, nElems*sizeof(T), cudaMemcpyHostToDevice));
    }

    static void copyDeviceToHost(T* host_dst, T* device_src, size_t nElems) {
    	gpuErrchk(cudaMemcpy(host_dst, device_src, nElems*sizeof(T), cudaMemcpyDeviceToHost));
	}

    static T* allocateHost(size_t nElems) {
    	return new T[nElems];
    }

    static T* allocateBytesHost(size_t nBytesClaimed) {
    	size_t nElems = nBytesClaimed/(sizeof(T));

    	if(nBytesClaimed % sizeof(T) != 0){
    		std::cerr << "Warning: non uniform allocation!" << std::endl;
    	}

    	T* result = new T[nElems];
#if _HC_OP_LOG == 1
    	size_t h_usBefore = h_us;
    	h_us += nBytesClaimed;
    	std::cout << "Host allocation: claimed: " <<
    			MemoryOperations::getMegs(nBytesClaimed) << "M, " <<
    			MemoryOperations::getMegs(h_usBefore) << "M -> " <<
    			MemoryOperations::getMegs(h_us) << "M" << endl;
#endif
    	return result;
    }

    static void releaseBytesHost(T* ptr, size_t nBytesReleased){
    	delete ptr;
#if _HC_OP_LOG == 1
    	size_t h_usBefore = h_us;
    	h_us -= nBytesReleased;

    	std::cout << "Host released: " <<
    			MemoryOperations::getMegs(nBytesReleased) << "M, " <<
    			MemoryOperations::getMegs(h_usBefore) << "M -> " <<
    			MemoryOperations::getMegs(h_us) << "M" << endl;
#endif

    }


    static T* allocateDevice(size_t nElems) {
    	T* result;
    	gpuErrchk( cudaMalloc((void**)&result, nElems * sizeof(T)));
    	return result;
	}

    static void printMemory(){
#if _DC_OP_LOG_CUDA == 1
    	size_t fr, to;
    	cuMemGetInfo(&fr, &to);
    	std::cout << "CUDA device stats: free: " << float(fr)/1024/1024 << "M total: " << float(to)/1024/1024 << "M << used: " <<  float(to-fr)/1024/1024 << "M" << std::endl;
#endif
    }

    static T* allocateBytesDevice(size_t nBytesClaimed) {
    	T* result;

#if _DC_OP_LOG == 1
    	std::cout << "Trying to allocate: " <<  getMegs(nBytesClaimed) << "M, " << nBytesClaimed << "B" << std::endl;
#endif

    	printMemory();
    	gpuErrchk( cudaMalloc((void**)&result, nBytesClaimed));
    	printMemory();

#if _DC_OP_LOG == 1
    	long usBefore = us;
    	us+=nBytesClaimed;

    	printf("Done. (%p)\n", result);
    	std::cout << "DeviceChunk stats (alloc): claimed: " <<
    			MemoryOperations::getMegs(nBytesClaimed) << "M, " <<
    			MemoryOperations::getMegs(usBefore) << "M expanded to " <<
    			MemoryOperations::getMegs(us) << "M" << endl;
#endif
    	return result;
	}

    static void releaseBytesDevice(T* ptr, size_t nBytesReleased){
#if _DC_OP_LOG == 1
    	std::cout << "Trying to release at: ";
    	printf("%p\n", ptr);
#endif

    	printMemory();
    	gpuErrchk(cudaFree(ptr));
    	printMemory();

#if _DC_OP_LOG == 1
    	size_t usBefore = us;
    	us-=nBytesReleased;

    	std::cout << "Done." << std::endl;
    	std::cout << "DeviceChunk stats (release): released: " <<
    			MemoryOperations::getMegs(nBytesReleased) << "M, " <<
    			MemoryOperations::getMegs(usBefore) << "M shrinked to " <<
    			MemoryOperations::getMegs(us) << "M" << endl;
#endif
    }
};

/**
 * Object that manages contiguous memory blocks on the host.
 * The constructor allocates the resource while the destructor releases.
 */
template<typename T>
class HostChunk : public Chunk<T> {
protected:
public:
	using uptr_t = unique_ptr<HostChunk<T>>;

	/**
	 * Calls the constructor with the same parameters and returns
	 * an unique_ptr to the newly created object.
	 */
	static uptr_t make_uptr(int nElems){
		return unique_ptr<HostChunk<T>>(new HostChunk<T>(nElems));
	}

	/**
	 * Calls the constructor with the same parameters and returns
	 * an unique_ptr to the newly created object.
	 */
	static uptr_t make_uptr(DeviceChunk<T>& other){
		return unique_ptr<HostChunk<T>>(new HostChunk<T>(other));
	}

	// Constructs a null host chunk with size 0, and a null pointer.
	HostChunk() : Chunk<T>(0, nullptr){}

	// Construct a host chunk with size a_elements and allocates the memory at the same time.
	HostChunk(int a_elements);

	// Construct the host chunk with size a_elements. The memory is not allocated, the
	// object will manages the memory passed using the h_pointer.
	HostChunk(int a_elements, T *h_pointer);

	/**
	 * Construct a host chunk from using a device chunk.
	 * The memory required memory will be allocated, and the data will be copied from the device chunk to the host.
	 */
	HostChunk(DeviceChunk<T>& other);

	// Releases the managed memory.
	~HostChunk();

	__host__ void fill(T val);

	/**
	 * Copies the data from an other host to this object: memcpy(other->this)
	 */
	void copyFrom(HostChunk<T>& other);
	void copyFromN(HostChunk<T>& other, int start, int nElems);
	void copyFromN(T* other, int nElems);

    T& operator[] (const int index);
    const T& operator[] (const int index) const;

    // Low level functionalities
    static void copyHostToHost(T* dst, T* src, int nElems) {
    	memcpy(dst, src, nElems*sizeof(T));
    }

    static T* allocateHost(int nElems) {
    	return MemoryOperations<T>::allocateBytesHost(sizeof(T)*nElems);
    }
};

template<typename T>
class DeviceChunk : public Chunk<T> {
public:
	static long usage;

	using uptr_t = unique_ptr<DeviceChunk<T>>;

	static uptr_t make_uptr(int nElems){
		return uptr_t(new DeviceChunk<T>(nElems));
	}

	static uptr_t make_uptr(HostChunk<T>& other){
		return unique_ptr<DeviceChunk<T>>(new DeviceChunk<T>(other));
	}

	static uptr_t make_uptr(DeviceChunk<T>& other){
		return unique_ptr<DeviceChunk<T>>(new DeviceChunk<T>(other, 42));
	}

	DeviceChunk() : Chunk<T>(0, nullptr){};
	DeviceChunk(T val, bool unused);
	DeviceChunk(int a_elements);
	DeviceChunk(int a_elements, T *h_pointer);
	DeviceChunk(HostChunk<T>& hostChunk);
	DeviceChunk(DeviceChunk<T>& deviceChunk, bool unused);
	//DeviceChunk(DeviceChunk<T>& deviceChunk) = delete;

	~DeviceChunk();

	__host__ void fill(T val);

	// device->device
	// memcpy(other -> this)
	void copyFrom(DeviceChunk<T>& other);

	void copyFromDeviceToDeviceN(T* d_pointer, int offset, int count);
	void copyFromDeviceToDeviceN(DeviceChunk<T>& other, int count);

	// memcpy(this -> other)
	void copy(DeviceChunk<T>& other);

	// device->host
	void copyHost(T *h_pointer);
	void copyHost(HostChunk<T> &hostChunk);

	void copyHostN(T *h_pointer, int count);
	void copyHostN(T *h_pointer, int offset, int count);
	void copyHostN(HostChunk<T> &hostChunk, int offset, int count);
	void copyHostN(HostChunk<T> &hostChunk, int count);

	// host->device
	void copyDevice(T *h_pointer);
	void copyDevice(HostChunk<T> &hostChunk);

	void copyDeviceN(T *h_pointer, int count);
	void copyDeviceN(T *h_pointer, int offset, int count);
	void copyDeviceN(HostChunk<T> &hostChunk, int offset, int count);
	void copyDeviceN(HostChunk<T> &hostChunk, int count);

	void putVal(T data, int offset);
	T getVal(int offset);

	void putVal(T data);
	T getVal();

	thrust::device_ptr<T> getTPtr();
	thrust::device_ptr<T> tbegin();
	thrust::device_ptr<T> tend();

	T* begin();
	T* end();

	__host__ T getAt(T* p);
	__host__ void setAt(T* p, T val);

	__host__ T getAt(int index);
	__host__ void setAt(int index, T val);


    __device__ T& operator[] (const int index);
    __device__ const T& operator[] (const int index) const;
};

template<typename T>
long DeviceChunk<T>::usage = 0;

template<typename T>
class DeviceVal : public DeviceChunk<T> {
public:
	DeviceVal(T val);
};

template<typename T>
DeviceVal<T>::DeviceVal(T val) : DeviceChunk<T>(val, true){};

// Chunk

template<typename T>
T* Chunk<T>::getPtr() {
	return this->pointer;
}

template<typename T>
int Chunk<T>::getElements() {
	return this->elements;
}

template<typename T>
Func3D<T> Chunk<T>::funcView(Size3D dims){
	return Func3D<T>(dims, this->pointer);
}

template<typename T>
Func2D<T> Chunk<T>::funcView(Size2D dims){
	return Func2D<T>(dims, this->pointer);
}

template<typename T>
Func1D<T> Chunk<T>::funcView(int nelems){
	return Func1D<T>(nelems, this->pointer);
}

template<typename T>
Func1D<T> Chunk<T>::funcView(){
	return Func1D<T>(this->getElements(), this->pointer);
}

template<typename T>
void Chunk<T>::rangeCheck(int index){
	if(index < 0 || index > elements-1){
#if defined(__CUDA_ARCH__)
		printf("Range error - array index out of bounds!");
#else
		throw range_error("Array index out of bounds!");
#endif
	}
}

// HostChunk

template<typename T>
HostChunk<T>::HostChunk(int a_elements) : Chunk<T>(a_elements){
	this->pointer = allocateHost(a_elements);
#if _HC_LOG == 1
	printf("HostChunk: %lu bytes allocated on the host at (host) %p.\n", a_elements * sizeof(T), this->pointer);
#endif
}

// TODO: wrong behaviour!

template<typename T>
HostChunk<T>::HostChunk(int a_elements, T *h_pointer) {
	cerr << "Unexpected behaviour! This function will be refactored to take over an unmanaged pointer instead of copying data from the provided area!" << endl;
	this->elements = a_elements;
	this->pointer = allocateHost(a_elements);
	copyHostToHost(this->pointer, h_pointer, a_elements);
#if _HC_LOG == 1
	printf("HostChunk: %lu bytes allocated on the host at %p.\n", a_elements * sizeof(T), this->pointer);
#endif
}

template<typename T>
HostChunk<T>::HostChunk(DeviceChunk<T>& other) {
	this->elements = other.getElements();
	this->pointer = allocateHost(other.getElements());
	cudaMemcpy(this->pointer, other.getPtr(), other.getElements()*sizeof(T), cudaMemcpyDeviceToHost);
#if _HC_LOG == 1
	printf("HostChunk: copying %lu bytes from the device to the host %p.\n", other.getElements() * sizeof(T), this->pointer);
#endif
}


template<typename T>
HostChunk<T>::~HostChunk() {
	if (this->elements > 0) {
	#if _HC_LOG == 1
		printf("HostChunk: Freeing %lu bytes on the host at %p\n", this->elements*sizeof(T), this->pointer);
	#endif
	
		MemoryOperations<T>::releaseBytesHost( this->pointer, this->elements*sizeof(T));
	
	#if _HC_LOG == 1
		printf("HostChunk: %lu bytes freed on the host at %p.\n", this->elements * sizeof(T), this->pointer);
	#endif
	}
}

template<typename T>
void HostChunk<T>::copyFrom(HostChunk<T>& other) {
	if(other.elements != this->elements){
		throw range_error(
				string("Range error during copy: (host) other -> (host) this. Capacity of other: " +
						to_string(other.elements) +
						" capacity of *this: " +
						to_string(this->elements)));
	}
	copyFromN(other.pointer, other.elements);
}

template<typename T>
void HostChunk<T>::copyFromN(HostChunk<T>& other, int start, int nElems) {
	copyFromN(other.getPtr() + start, nElems);
}

template<typename T>
void HostChunk<T>::copyFromN(T* other, int nElems) {
	if(nElems > this->elements){
		throw range_error(
				string("Range error during copy: (host) other -> (host) this. Number of elements to copy (nElems): " +
						to_string(nElems) +
						" capacity of *this: " +
						to_string(this->elements)));
	}
	copyHostToHost(this->pointer, other, nElems);
}

template<typename T>
T& HostChunk<T>::operator[] (const int index){
//	this->rangeCheck(index);
    return (this->pointer)[index];
}

template<typename T>
const T& HostChunk<T>::operator[] (const int index) const {
//	this->rangeCheck(index);
	return (this->pointer)[index];
}

template<typename T>
__host__ void HostChunk<T>::fill(T val){
	thrust::fill(thrust::host, this->pointer, this->pointer+this->elements, val);
}

// DeviceChunk

template<typename T>
DeviceChunk<T>::DeviceChunk(T val, bool unused) : Chunk<T>(1) {
	int nElems = 1;
	size_t nBytesClaimed = nElems * sizeof(T);
	this->pointer = MemoryOperations<T>::allocateBytesDevice(nBytesClaimed);
	putVal(val);
#if _DC_LOG == 1
	printf("DeviceChunk(T val): %d bytes allocated on the device at %p. Current usage: %f M\n", nBytesClaimed, this->pointer, MemoryOperations<T>::getMegs(us));
#endif
}


template<typename T>
DeviceChunk<T>::DeviceChunk(int a_elements) : Chunk<T>(a_elements) {
	size_t nBytesClaimed = a_elements*sizeof(T);
#if _DC_LOG == 1
	std::cout << "DeviceChunk(int): " << nBytesClaimed << " bytes" << std::endl;
#endif
	this->pointer = MemoryOperations<T>::allocateBytesDevice(nBytesClaimed);
#if _DC_LOG == 1
	printf("DeviceChunk(nElements): %d bytes allocated on the device at %p. Current usage: %f M\n", nBytesClaimed, this->pointer, MemoryOperations<T>::getMegs(us));
#endif
}

template<typename T>
DeviceChunk<T>::DeviceChunk(int a_elements, T *h_pointer) {
	this->elements = a_elements;
	size_t nBytesClaimed = a_elements * sizeof(T);
	this->pointer = MemoryOperations<T>::allocateBytesDevice(nBytesClaimed);
	gpuErrchk( cudaMemcpy(this->pointer, h_pointer, nBytesClaimed, cudaMemcpyHostToDevice) );
#if _DC_LOG == 1
	printf("DeviceChunk(nElements, hostPtr): %d bytes copied to the device. Current usage: %f M\n", nBytesClaimed, MemoryOperations<T>::getMegs(us));
#endif
}

template<typename T>
DeviceChunk<T>::DeviceChunk(HostChunk<T>& hostChunk) {
	size_t hostNElements = hostChunk.getElements();
	this->elements = hostNElements;
	size_t nBytesClaimed = hostNElements * sizeof(T);
	this->pointer = MemoryOperations<T>::allocateBytesDevice(nBytesClaimed);
	gpuErrchk( cudaMemcpy(this->pointer, hostChunk.getPtr(), nBytesClaimed, cudaMemcpyHostToDevice) );
#if _DC_LOG == 1
	printf("DeviceChunk(hostChunk): %d bytes copied to the device. Current usage: %f M\n", nBytesClaimed, MemoryOperations<T>::getMegs(us));
#endif
}

template<typename T>
DeviceChunk<T>::DeviceChunk(DeviceChunk<T>& deviceChunk, bool unused) {
	size_t deviceNElements = deviceChunk.getElements();
	this->elements = deviceNElements;
	size_t nBytesClaimed = deviceNElements * sizeof(T);
	this->pointer = MemoryOperations<T>::allocateBytesDevice(nBytesClaimed);
	gpuErrchk( cudaMemcpy(this->pointer, deviceChunk.getPtr(), nBytesClaimed, cudaMemcpyDeviceToDevice) );
#if _DC_LOG == 1
	printf("DeviceChunk(deviceChunk): %d bytes copied on the device from src to dst. Current usage: %f M\n", nBytesClaimed, MemoryOperations<T>::getMegs(us));
#endif
}

template<typename T>
DeviceChunk<T>::~DeviceChunk() {
	if (this->elements > 0) {
#if _DC_LOG == 1
		printf("DeviceChunk: Freeing %lu bytes on the device at %p\n", this->elements*sizeof(T), this->pointer);
#endif
		size_t releasedSize = this->elements*sizeof(T);
		MemoryOperations<T>::releaseBytesDevice(this->pointer, releasedSize);
#if _DC_LOG == 1
		printf("DeviceChunk %lu bytes freed on the device at %p. Current usage: %f M\n", this->elements * sizeof(T), this->pointer, MemoryOperations<T>::getMegs(us));
#endif
	}
}

// Device to host

template<typename T>
void DeviceChunk<T>::copyHost(T *h_pointer) {
	gpuErrchk( cudaMemcpy(h_pointer, this->pointer, this->elements * sizeof(T), cudaMemcpyDeviceToHost) );
}

template<typename T>
void DeviceChunk<T>::copyHost(HostChunk<T>& hostChunk) {
	copyHost(hostChunk.getPtr());
}

// Device to host N

template<typename T>
void DeviceChunk<T>::copyHostN(T *h_pointer, int count) {
	copyHostN(h_pointer, 0, count);
}

template<typename T>
void DeviceChunk<T>::copyHostN(HostChunk<T>& hostChunk, int offset, int count) {
	copyHostN(hostChunk.getPtr(), offset, count);
}

template<typename T>
void DeviceChunk<T>::copyHostN(HostChunk<T>& hostChunk, int count) {
	copyHostN(hostChunk.getPtr(), 0, count);
}

template<typename T>
void DeviceChunk<T>::copyHostN(T *h_pointer, int offset, int count) {
#if _DC_LOG == 1
		printf("Copying %lu bytes from device to host: %p->%p\n", count * sizeof(T), (this->pointer)+offset, h_pointer);
#endif
	gpuErrchk(cudaMemcpy(h_pointer, (this->pointer)+offset, count * sizeof(T), cudaMemcpyDeviceToHost));
}

// ...

template<typename T>
thrust::device_ptr<T> DeviceChunk<T>::getTPtr() {
	return thrust::device_ptr<T>(this->pointer);
}

template<typename T>
thrust::device_ptr<T> DeviceChunk<T>::tbegin() {
	return thrust::device_ptr<T>(this->pointer);
}

template<typename T>
thrust::device_ptr<T> DeviceChunk<T>::tend() {
	return thrust::device_ptr<T>(this->pointer+this->elements);
}

template<typename T>
T* DeviceChunk<T>::begin() {
	return this->pointer;
}

template<typename T>
T* DeviceChunk<T>::end() {
	return this->pointer+this->elements;
}

template<typename T>
T DeviceChunk<T>::getAt(T* p) {
	T val;
	MemoryOperations<T>::copyDeviceToHost(&val, p, 1);
	return val;
}

template<typename T>
void DeviceChunk<T>::setAt(T* p, T val) {
	MemoryOperations<T>::copyHostToDevice(p, &val, 1);
}

template<typename T>
T DeviceChunk<T>::getAt(int index) {
	T val;
	MemoryOperations<T>::copyDeviceToHost(&val, &(this->pointer[index]), 1);
	return val;
}

template<typename T>
void DeviceChunk<T>::setAt(int index, T val) {
	MemoryOperations<T>::copyHostToDevice(&(this->pointer[index]), &val, 1);
}

// ...


// Host -> device

template<typename T>
void DeviceChunk<T>::copyDevice(T* h_pointer) {
	gpuErrchk( cudaMemcpy(this->pointer, h_pointer, this->elements * sizeof(T), cudaMemcpyHostToDevice) );
}

template<typename T>
void DeviceChunk<T>::copyDevice(HostChunk<T>& hostChunk) {
	copyDevice(hostChunk.getPtr());
}

// Host -> device N

template<typename T>
void DeviceChunk<T>::copyDeviceN(T* h_pointer, int count) {
	copyDeviceN(h_pointer, 0, count);
}

template<typename T>
void DeviceChunk<T>::copyDeviceN(T* h_pointer, int offset, int count) {
	gpuErrchk(cudaMemcpy((this->pointer)+offset, h_pointer, count * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void DeviceChunk<T>::copyDeviceN(HostChunk<T>& hostChunk, int offset, int count) {
	copyDeviceN(hostChunk.getPtr(), offset, count);
}

template<typename T>
void DeviceChunk<T>::copyDeviceN(HostChunk<T>& hostChunk, int count) {
	copyDeviceN(hostChunk.getPtr(), 0, count);
}

// ...

template<typename T>
void DeviceChunk<T>::putVal(T data) {
	copyDeviceN(&data, 1);
}

template<typename T>
void DeviceChunk<T>::putVal(T data, int offset) {
	copyDeviceN(&data, offset, 1);
}

template<typename T>
T DeviceChunk<T>::getVal() {
	T data;
	copyHostN(&data, 1);
	return data;
}

template<typename T>
T DeviceChunk<T>::getVal(int offset) {
	T data;
	copyHostN(&data, offset, 1);
	return data;
}

// Device -> device

// Other -> this

template<typename T>
void DeviceChunk<T>::copyFromDeviceToDeviceN(T* d_pointer, int offset, int count){
	gpuErrchk(cudaMemcpy((this->pointer)+offset, d_pointer, count * sizeof(T), cudaMemcpyDeviceToDevice));
}

template<typename T>
void DeviceChunk<T>::copyFromDeviceToDeviceN(DeviceChunk<T>& other, int count){
	copyFromDeviceToDeviceN(other.getPtr(), 0, count);
}

template<typename T>
void DeviceChunk<T>::copyFrom(DeviceChunk<T>& other){
	if(other.elements != this->elements){
		throw range_error("Error during copying data from data on the device. The chunks have different size!");
	}
	cudaMemcpy(this->pointer, other.pointer, this->pointer*sizeof(T), cudaMemcpyDeviceToDevice);
}

template<typename T>
void DeviceChunk<T>::copy(DeviceChunk<T>& other){
	if(other.elements != this->elements){
		throw range_error("Error during copying data from data on the device. The chunks have different size!");
	}
	cudaMemcpy(other.pointer, this->pointer, other.elements*sizeof(T), cudaMemcpyDeviceToDevice);
}

template<typename T>
__device__ T& DeviceChunk<T>::operator[] (const int index){
	//this->rangeCheck(index);
    return (this->pointer)[index];
}

template<typename T>
__device__ const T& DeviceChunk<T>::operator[] (const int index) const {
	//this->rangeCheck(index);
	return (this->pointer)[index];
}

template<typename T>
__host__ void DeviceChunk<T>::fill(T val){
	thrust::fill(thrust::device, thrust::device_ptr<T>(this->pointer), thrust::device_ptr<T>(this->pointer)+this->elements, val);
}


template<class T>
using chunk = Chunk<T>;

template<class T>
using dchunk = DeviceChunk<T>;

template<class T>
using hchunk = HostChunk<T>;

// floating point

using hchunk_float = hchunk<float>;
using dchunk_float = dchunk<float>;

// Integers, default

// signed

using hchunk_int = hchunk<int>;
using dchunk_int = dchunk<int>;

using hchunk_short = hchunk<short>;
using dchunk_short = dchunk<short>;

using hchunk_char = hchunk<char>;
using dchunk_char = dchunk<char>;

// Integers, fixed width

// signed

using hchunk_int32 = hchunk<int32_t>;
using dchunk_int32 = dchunk<int32_t>;

using hchunk_int16 = hchunk<int16_t>;
using dchunk_int16 = dchunk<int16_t>;

using hchunk_int8 = hchunk<int8_t>;
using dchunk_int8 = dchunk<int8_t>;

// unsigned

using hchunk_uint32 = hchunk<uint32_t>;
using dchunk_uint32 = dchunk<uint32_t>;

using hchunk_uint16 = hchunk<uint16_t>;
using dchunk_uint16 = dchunk<uint16_t>;

using hchunk_uint8 = hchunk<uint8_t>;
using dchunk_uint8 = dchunk<uint8_t>;

// Old API

typedef DeviceChunk<float> DevFloatChk;
typedef DeviceChunk<int> DevIntChk;
typedef DeviceChunk<bool> DevBoolChk;
typedef DeviceChunk<uint8_t> DevUByteChk;
typedef DeviceChunk<uint16_t> DevUShortChk;
typedef DeviceChunk<uint32_t> DevUIntChk;
typedef DeviceChunk<IntPair> DevIntPairChk;

typedef HostChunk<float> HostFloatChk;
typedef unique_ptr<HostFloatChk> HostFloatChkUPtr;

typedef HostChunk<int> HostIntChk;
typedef HostChunk<bool> HostBoolChk;
typedef HostChunk<uint8_t> HostUByteChk;
typedef HostChunk<uint16_t> HostUShortChk;
typedef HostChunk<uint32_t> HostUIntChk;
typedef HostChunk<IntPair> HostIntPairChk;

#endif
