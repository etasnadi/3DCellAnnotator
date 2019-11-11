#ifndef QMITKASTOOLGUI_H_
#define QMITKASTOOLGUI_H_

#include "QmitkToolGUI.h"
#include "mitkAsTool.h"
#include "SimpleConfig.cuh"
#include <MitkSegmentationUIExports.h>
#include <qfuturewatcher.h>
#include <qpushbutton.h>
#include <qcheckbox.h>
#include <qthread.h>
#include <qlineedit.h>
#include <vector>

// QT gui
#include <qgroupbox.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qpushbutton.h>
#include <QDoubleSpinBox>
#include <QSlider>
#include <qradiobutton.h>
#include <qlineedit.h>
#include <qfuture.h>
#include <QtConcurrent>


class CusSlider;
class SegRunner;

class MITKSEGMENTATIONUI_EXPORT QmitkAsToolGUI : public QmitkToolGUI {

	Q_OBJECT

public:
	mitkClassMacro(QmitkAsToolGUI, QmitkToolGUI);
	itkFactorylessNewMacro(Self) itkCloneMacro(Self)
	static SimpleConfig configUpdates;
	static SimpleConfig configInit;
	static int k;
	static float spherW;
	static map<std::string, double> savedW;

signals:
	void Init(mitk::Image *im, mitk::Image *seg, SimpleConfig conf, int labelId);
	void Cleanup();
	void DoSegStep(SimpleConfig conf);


public slots:
	void OnStartButtonClicked();
	void OnStopButtonClicked();

	void OnReportNewSurface(p_ObjectStat);

	void OnAcceptSegmentation();
	void OnAcceptSagitalSlice();

	void OnNewToolAssociated(mitk::Tool *);

protected:
	QmitkAsToolGUI();
	~QmitkAsToolGUI();

private:
	mitk::AsTool::Pointer m_asTool;
	QThread workerThread;
	SegRunner* runner;

	// Segmentation control
	int getSelectedLabelId(mitk::Image* image);
	void launchNextStep(SimpleConfig conf);
	void startSegmentation();
	void stopSegmentation();

	// Controls the segmentation status
	bool segRunning;
	int nIterationsElapsed;
	int currentWorkingLabelId;
	bool autoJumpNext;
	int autoJumpLimit;
	bool activeSession;

	// Gui elements
	void buildGUI();
	QPushButton *startSegButton;
	QPushButton *stopSegButton;
	QLineEdit *actualVolumeVal;
	QLineEdit *actualPlasmaVal;
	CusSlider *plasmaTargetCSldr;
	CusSlider *volumeTargetCSldr;
	QCheckBox *initFromSphere;

	// Rendering objects
	mitk::DataNode::Pointer m_surfaceDataNode;


	void initEvolutionVisualizationSurface();
	void updateOrAddEvolutionVisualization(mitk::Surface::Pointer a_surface);
	void applyCurrentSegmentation();
	mitk::Surface::Pointer getSurfaceFromSegmentation(float* surfData,
		int w, int h, int d,
		int tx, int ty, int tz);
	void removeEvolutionVisualization();

	// Statics:

	QDoubleSpinBox* createSpinBoxParam(SimpleConfig& configInit, std::string paramName);
	QCheckBox* createToggleParam(std::string label, SimpleConfig& configInit, std::string paramName);

	static void showDataInfo(mitk::ToolManager *toolManager);
	static mitk::Image::Pointer getSelectedInitializer(mitk::ToolManager *toolManager);
	bool activateNextLabe();
	static mitk::Image::Pointer getBinaryMaskFromSurface(mitk::Surface *surf, bool segTypeUshort, mitk::Image *refImage);

	// Other:

	void readConfig();

};

class SegRunner : public QObject {
	Q_OBJECT

signals:
	void ReportNewSurface(p_ObjectStat);

public:
	SegRunner(mitk::AsTool::Pointer a_asTool) : m_asTool(a_asTool){};
	~SegRunner(){};

public slots:
	void OnDoSegStep(SimpleConfig conf);
	void OnInit(mitk::Image *im, mitk::Image *seg, SimpleConfig conf, int labelId);
	void OnCleanup();

private:
	int iter;
	bool seg;
	mitk::AsTool::Pointer m_asTool;
};

class CusSlider : public QFrame {
	Q_OBJECT
private:
	QGridLayout *sliderGbLay;
	QSlider *slider;
	std::vector<QLabel*> labels;
	int nLabels;
	int resolution;
	std::pair<float, float> interval;
public:
	CusSlider(int a_nLabels, QWidget *parent = 0);
	void updateIv(int resolution, std::pair<float, float> a_interval);
	QSlider *getSlider();
	float translateVal(int v);
	~CusSlider();
};

#endif /* QMITKASTOOLGUI_H_ */
