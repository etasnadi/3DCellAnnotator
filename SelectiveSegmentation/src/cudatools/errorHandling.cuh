#pragma once

#include <stdio.h>

////////////////////////////////////////////////////////////////////////////////
// Cuda error checking
////////////////////////////////////////////////////////////////////////////////

void SAFE_CALL(cudaError_t err);

void KERNEL_ERROR_CHECK();
void KERNEL_ERROR_CHECK(char const *message);

void GL_ERROR_CHECK();