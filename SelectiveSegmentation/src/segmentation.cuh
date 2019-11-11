#ifndef SEGMENTATION_CUH
#define SEGMENTATION_CUH 1

#include <mutex>
#include "common.cuh"
#include "surfaceAndVolume.cuh"

class SegContext {
private:
	shared_ptr<CurveProps> objectStats;

	// Input
	GridParams gridParams;
	AlgParams algParams;
	Obj preferredObject;

	HostChunk<uint16_t>* inputImage;
	Size3D imageDims;
	int borderSize;

	int3 gridToImageTranslation;

	// Working data structures
	unique_ptr<DevFloatChk> workImage;
	unique_ptr<DevIntChk> closestObj;
	unique_ptr<DevFloatChk> auxField;
	unique_ptr<DevFloatChk> dataTerm;
	unique_ptr<DeviceChunk<FltDer3D>> imgDer;
	dchunk<float3>::uptr_t normals;
	dchunk<float>::uptr_t K;

	unique_ptr<DevIntPairChk> ccGraph;
	unique_ptr<DevIntChk> ccResult;

	bool graphComputed = false;

	// State
	int iterations = 0;
	//Size3D gridSize;
	std::vector<Obj> objectSummary;
	int nComps_sh = 0;
	int evolutionStrategy = 0;

	/*
	 * Performs an evolution step.
	 */
	void evolve();

	/*
	 * Regularises the level set before updating it.
	 */
	void regularise();

	void initAlg();

	void preprocessImage();

	/*
	 * Computes the things that are unchanged during the evolution.
	 */
	void computeData();

	/*
	 * Initializes the inner data structures needed to the level set evolution and some metadata like the grid size.
	 */
	void initImageStructures(int imagePts); // Initializes the image related structures (usually they are not changed during the evolution)
	void initLevelSetStructures(int gridPts);	// Initializes the grid related structures
	void initGridAuxStructures(int gridPts);

	void updateGridStructures(int gridPts);
	/*
	 * Initializes the level set with a default object like a sphere or a cube based on the configuration parameters.
	 */
	void initLevelSet();

	/*
	 * Call with this function if you want to manually initialize the level set using a mask.
	 * This will called from the constructor if a mask is passed to this object.
	 * The mask should contain value 1 inside the contour and 0 outside.
	 */
	void initLevelSetWithMask(hchunk<uint8_t>& mask);

	/**
	 * Checks the extrema of the mask and initializes the level set with the minimal enclosing sphere.
	 */
	void initLevelSetWithEnclosingSphere(hchunk<uint8_t>& mask);
public:
	float lastVol;
	unique_ptr<DevFloatChk> ls1;
	unique_ptr<DevFloatChk> ls2;

	// TODO: Just for debugging!
	Point3D phaseIntLine;
	int getObjSummary();
	/**
	 * Initializes the segmentation.
	 * The image will be copied to the device,
	 * the data term will be computed and copied to the device,
	 * the memory is alloceted for the structures.
	 */
	SegContext(HostFloatChk& image, Size3D aImageDims, int aBorderSize, AlgParams aAlgParams, GridParams& aGridProps, Obj aPrefObj, int strategy, int evolutionStrategy, HostUByteChk* mask = nullptr);

	/**
	 * The same as the previous contructor, but with an extra parameter initLevelSet
	 */
	//SegContext(HostFloatChk& image, HostFloatChk& initLevelSet, Size3D aImageDims, int aBorderSize, AlgParams aAlgParams, GridParams& aGridProps, Obj aPrefObj);

	/*
	 * Rescales the level set to the minimum size needed to maintain the contour.
	 *
	 * If the contour grows too big then the grid size will be increased, otherwise
	 * it won't be able to maintain the level set.
	 * If the grid is much bigger than the contour, it will cut the grid to have the minimal size
	 * that is still enough to perform the level set evolution.
	 *
	 * The new size of the level set is returned for convinience.
	 *
	 * This function is called before every iteration.
	 */
	Size3D resizeOptimalIfNeeded();

	void iterate();
	DevFloatChk& getLevelSet();
	unique_ptr<HostFloatChk> getLevelSetView();
	unique_ptr<HostIntChk> getObjectIds();
	DevFloatChk& getImage();
	int getIterations();
	void updateAlgParams(AlgParams aAlgParams);
	GridParams getGridParams();
	void updatePrior(Obj prior);
	void loadLevelSet(HostFloatChk& aLevelSet);
	unique_ptr<HostIntPairChk> getCCgraph();
	unique_ptr<HostIntChk> getCCResult();
	unique_ptr<HostUByteChk> getBinarySegmentation();
	int3 getGridToImageTranslation();

	shared_ptr<CurveProps> getObjects();
};

#endif
