#ifndef SELECTIVE_H
#define SELECTIVE_H

#include "macros.h"
#include <tuple>
#include "SimpleConfig.cuh"

typedef struct p_ObjectStat {
	float vol;
	float surf;
	float plasma;
	int gw, gh, gd;
} p_ObjectStat;

typedef struct p_int3 {
	int x;
	int y;
	int z;
} p_int3;

/**
 * Initializes the algorithm.
 * image: an unsigned integer image, each pixel is an eight bit gray value.
 * segmentation: an unsigned integer image, each pixel is a 16-bit label id. It is the level set initializer depending on the configuration.
 * labelId: selects the object provided in the previous label set image.
 * pixelSize: unused
 * imageSize: the dimensions of the image and the segmentation
 * a_conf: the configuration
 */
EXPORT_SHARED int segmentation_app_headless_init(const void* image, const void* segmentation, int labelId, int pixelSize, p_int3 imageSize, SimpleConfig a_conf);

/**
 * Performs a simple evolution step. The configuration will be updated with the one passed in the argument.
 * Returns an object statistics.
 */
EXPORT_SHARED p_ObjectStat segmentation_app_headless_step(SimpleConfig conf);

/*
 * Returns the current level set as a float array. The level set size will be placed in the output variable gsize, and the translation will be put to the trans variable.
 * The algorithm only maintains a level set where there is an object. The frame will be extracted and may be updated during the segmentation. The gsize variable tells the
 * client the frame size, while the trans variable holds the translation to the image origin to the frame origin.
 */
EXPORT_SHARED float* segmentation_app_grab_level_set(p_int3& gsize, p_int3& trans);

/*
 * Releases the resources allocated.
 */
EXPORT_SHARED int segmentation_app_headless_cleanup();

#endif
