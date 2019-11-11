/*
 * mitkAsTool.h
 *
 *  Created on: 2018 dec. 29
 *      Author: Ervin Áron Tasnádi
 */

#ifndef mitkAsTool_h_Included
#define mitkAsTool_h_Included

#include "mitkCommon.h"
#include "mitkAsToolInterface.h"
#include "mitkFeedbackContourTool.h"
#include <MitkSegmentationExports.h>
 #include <mitkAutoSegmentationTool.h>

#include "mitkSurface.h"
#include "selective.h"

#include "SimpleConfig.cuh"

#include <mitkInteractionEvent.h>
#include "mitkStateMachineAction.h"

#define mitkAsTool_NAME "3D Cell Annotator"
#define mitkAsTool_DESCRIPTION "3D Cell Annotator _"
//#define mitkAsTool_ICON "o1g-w.png"
#define mitkAsTool_ICON "AS.png"

namespace mitk {

	//class MITKSEGMENTATION_EXPORT AsTool : public FeedbackContourTool {
	class MITKSEGMENTATION_EXPORT AsTool : public AutoSegmentationTool {
	public:
		void ConnectActionsAndFunctions() override;

		virtual void OnMousePressed(StateMachineAction *, InteractionEvent *interactionEvent);

		//mitkClassMacro(AsTool, FeedbackContourTool);
		mitkClassMacro(AsTool, AutoSegmentationTool);
		itkFactorylessNewMacro(Self) itkCloneMacro(Self)
		us::ModuleResource GetIconResource() const override;
		const char **GetXPM() const override;
		const char *GetName() const override;

		// Controlling the segmentation
		void init(mitk::Image *im, mitk::Image *seg, SimpleConfig conf, int labelId);
		p_ObjectStat launchNextStepComputation(SimpleConfig conf);
		static float* getNextStepLevelSet(p_int3& gridSize, p_int3& translation);
		void cleanup();
		ToolManager *getToolManager();
	protected:
		AsTool();             // purposely hidden
		AsTool(const char *); // purposely hidden
		~AsTool() override;
	private:
		void initSegSurface();
	};

}


#endif /* mitkAsTool */
