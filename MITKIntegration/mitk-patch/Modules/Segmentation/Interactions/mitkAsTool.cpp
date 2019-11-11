#include "mitkAsTool.h"
#include "mitkFeedbackContourTool.h"
#include "mitkDataStorage.h"
#include "mitkToolManager.h"
#include "mitkRenderingManager.h"
// Interface
#include "mitkAsToolInterface.h"
#include "selective.h"

#include <MitkSegmentationExports.h>
#include <mitkAutoSegmentationTool.h>
#include "mitkImage.h"
#include "mitkSurfaceToImageFilter.h"
#include "itkImageIOBase.h"
#include <mitkImageReadAccessor.h>
#include "mitkStateMachineAction.h"
#include "mitkInteractionEvent.h"
#include <mitkInteractionPositionEvent.h>
#include "mitkLabelSetImage.h"

#include <usGetModuleContext.h>
#include <usModule.h>
#include <usModuleContext.h>
#include <usModuleResource.h>
#include <vtkImageData.h>
#include <vtkImageMathematics.h>

#include <iostream>

namespace mitk {
 	 MITK_TOOL_MACRO(MITKSEGMENTATION_EXPORT, AsTool, mitkAsTool_DESCRIPTION);
}


void mitk::AsTool::ConnectActionsAndFunctions(){
  CONNECT_FUNCTION("PrimaryButtonPressed", OnMousePressed);
}

void mitk::AsTool::OnMousePressed(StateMachineAction *, InteractionEvent *interactionEvent){

	auto *positionEvent = dynamic_cast<mitk::InteractionPositionEvent *>(interactionEvent);
	if (!positionEvent){
		return;
	}
}

//mitk::AsTool::AsTool() : FeedbackContourTool("PressMoveReleaseWithCTRLInversion") {
mitk::AsTool::AsTool(){
}

mitk::AsTool::~AsTool(){
}

// Controlling the segmentation
void mitk::AsTool::init(mitk::Image* im, mitk::Image *seg, SimpleConfig conf, int labelId){
	// Acquiring the image...
	int nDims = im->GetDimension();
	for(int i = 0; i < nDims; i++){
		std::cout << "Dims (image): (" << i << "): " << im->GetDimension(i) << std::endl;
		std::cout << "Dims: (segmentation) (" << i << "): " << seg->GetDimension(i) << std::endl;
	}

	int nComponents = im->GetPixelType().GetNumberOfComponents();
	int pixelSize = im->GetPixelType().GetSize();


	std::cout << "Image: pixel components: " << nComponents << ", pixel size: " << pixelSize << std::endl;

	ImageReadAccessor imageAcc(im);
	const void *imagePtr = imageAcc.GetData();

	p_int3 siz;
	siz.x = im->GetDimension(0);
	siz.y = im->GetDimension(1);
	siz.z = im->GetDimension(2);

	ImageReadAccessor segAcc(seg);
	const void *segPtr = segAcc.GetData();
	int nSegComponents = seg->GetPixelType().GetSize();
	std::cout << "Segmentation: pixel components: " << nSegComponents << std::endl;

	std::cout << "Current label id: " << labelId << std::endl;
	
	/**
	 * Level set initialization strategy:
	 * 0 -> using the provided mask
	 * 1 -> using a minimal enclosing sphere around the provided mask
	 * 2 -> using an object with the parameters given in the alg params.
	 */

	segmentation_app_headless_init(imagePtr, segPtr, labelId, pixelSize, siz, conf);
}

p_ObjectStat mitk::AsTool::launchNextStepComputation(SimpleConfig conf){
	return segmentation_app_headless_step(conf);
}

float* mitk::AsTool::getNextStepLevelSet(p_int3& gridSize, p_int3& translation){
	return segmentation_app_grab_level_set(gridSize, translation);
}

void mitk::AsTool::cleanup(){
	segmentation_app_headless_cleanup();
}


mitk::ToolManager *mitk::AsTool::getToolManager(){
	return m_ToolManager;
}

us::ModuleResource mitk::AsTool::GetIconResource() const {
	us::Module *module = us::GetModuleContext()->GetModule();
	us::ModuleResource resource = module->GetResource(mitkAsTool_ICON);
	return resource;
}

const char **mitk::AsTool::GetXPM() const {
	return nullptr;
}

const char *mitk::AsTool::GetName() const {
	return mitkAsTool_NAME;
}
