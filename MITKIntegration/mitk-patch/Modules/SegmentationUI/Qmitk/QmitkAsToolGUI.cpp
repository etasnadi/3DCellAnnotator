#include "QmitkAsToolGUI.h"

// std
#include <tuple>
#include <iostream>
//#include <unistd.h>
#include <vector>

#include <mitkToolManager.h>
#include "mitkTool.h"
#include "mitkAsTool.h"
#include "mitkLabelSetImage.h"

// QT gui
#include <qgroupbox.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qpushbutton.h>
#include <QDoubleSpinBox>
#include <QSlider>
#include <qradiobutton.h>
#include <qcheckbox.h>
#include <qlineedit.h>
#include <qfuture.h>
#include <QtConcurrent>
#include <QCheckBox>

#include "mitkSurface.h"
#include "mitkSurfaceToImageFilter.h"
#include <mitkImageReadAccessor.h>
#include <mitkImageWriteAccessor.h>

// VTK

#include "vtkSphereSource.h"
#include "vtkImageImport.h"
#include "vtkMarchingCubes.h"
#include "vtkTransformPolyDataFilter.h"
#include "vtkTransform.h"

#include <mitkIOUtil.h>

// Button labels
#define PAUSE_LABEL "Pause"
#define RESUME_LABEL "Resume"
#define START_SEG_LABEL "Start"
#define CANCEL_SEG_LABEL "Cancel"

// Config property names
#define VOLUME_WEIGHT_PROPERTY "eq.lambda"
#define PLASMA_WEIGHT_PROPERTY "eq.mu"
#define SMOOTHNESS_WEIGHT_PROPERTY "eq.theta"
#define DATA_WEIGHT_PROPERTY "eq.eta"

#define VOLUME_PRIOR_PROPERTY "pref.vol"
#define PLASMA_PRIOR_PROPERTY "pref.p"

#define SPHERE_AMOEBA 10.6

#define GUI_CONF_PREFIX "gui."
#define GUI_CONF_POSTFIX_ENABLED ".enabled"
#define GUI_CONF_POSTFIX_MUL ".mul"

#define SETTINGS_FILENAME "settings_3DCA.conf"

#define AUTO_JUMP_LIMIT_DEFAULT 50

#include "mitkSliceNavigationController.h"

// QmitkAsToolGUI

/*
 * The user selects the active surfaces tool, the mouse cursor will be changed to the selection icon.
 *
 * Initially no centroid is selected and the active surface properties is disabled except the initial contour type selection box.
 *
 * The user selects a poing by pushing the left mouse button somewhere in one of the planar views (or maybe even in the 3D view)
 * to select the centroid of an initial surface (that can be a sphere, cube, ellipsoid, etc.). Without releasing the button, the user moves
 * the cursor to adjust the size of the initial contour (e.g. radius of the sphere). When the user releases the button, the initial contour
 * stays on the image, and the properties page is enabled. If the user clicks again on the image, the previous contour is dropped
 *
 * Visually, there should be an active segmentation that has a marker color c.
 * When the user adjusts the centroid, the contours shown on all of the views with a color "light c".
 *
 * Then, the segmentation is started by pressing the start segmentation button or an accelerator (space). The start segmentation button is changed
 * to the pause segmentation button.
 *
 */

MITK_TOOL_GUI_MACRO(MITKSEGMENTATIONUI_EXPORT, QmitkAsToolGUI, "")

/**
 * CusSlider class
 */

CusSlider::CusSlider(int a_nLabels, QWidget *parent): QFrame(parent), nLabels(a_nLabels){
	sliderGbLay = new QGridLayout(this);

	slider = new QSlider(Qt::Horizontal);
	for(int i = 0; i < nLabels; i++){
		QLabel *currLabel = new QLabel();
		labels.push_back(currLabel);
		sliderGbLay->addWidget(currLabel, 1, i, 1, 1, Qt::AlignHCenter);
	}

	sliderGbLay->addWidget(slider, 0, 0, 1, nLabels);
}

void CusSlider::updateIv(int a_resolution, std::pair<float, float> a_interval){
	interval = a_interval;
	resolution = a_resolution;

	slider->setMinimum(0);
	slider->setMaximum(resolution);
	slider->setSliderPosition(resolution/2);
	slider->setSingleStep(1);
	slider->setTickPosition(QSlider::TicksBelow);

	float ivSize = interval.second - interval.first;

	for(int i = 0; i < nLabels; i++){
		float unitDist = ivSize/nLabels;
		float label = interval.first + unitDist*i + unitDist/2.0f;
		QLabel *lab = labels[i];
		lab->setText(QString::number(label));
	}
}

float CusSlider::translateVal(int v){
	return interval.first +(interval.second-interval.first)*1.0/float(resolution)*v;
}

QSlider* CusSlider::getSlider(){
	return slider;
}

CusSlider::~CusSlider(){
}

/**
 * QmitkAsToolGui class
 */

// Initial configuration.
SimpleConfig QmitkAsToolGUI::configInit;

// This instance will be used to communicate the updates to the algorithm.
SimpleConfig QmitkAsToolGUI::configUpdates;

// Saving the parameters to reuse if the tool is inactivated.
map<std::string, double> QmitkAsToolGUI::savedW;
int QmitkAsToolGUI::k;

QDoubleSpinBox* QmitkAsToolGUI::createSpinBoxParam(SimpleConfig& configInit, std::string paramName){
	string cRoot = GUI_CONF_PREFIX;
	QDoubleSpinBox *inp = new QDoubleSpinBox();

	double v = configInit.getFProperty(paramName);

	inp->setMinimum(configInit.getFProperty(cRoot+paramName+".min"));
	inp->setMaximum(configInit.getFProperty(cRoot+paramName+".max"));
	inp->setSingleStep(configInit.getFProperty(cRoot+paramName+".step"));
	
	if(savedW.count(paramName) > 0){
		v = savedW[paramName];
	}
	inp->setValue(v);
	string paramPropertyName = GUI_CONF_PREFIX + string(paramName) + GUI_CONF_POSTFIX_MUL;
	configUpdates.setFProperty(paramName, v*configInit.getFProperty(paramPropertyName));
	return inp;
}

QCheckBox* QmitkAsToolGUI::createToggleParam(std::string label, SimpleConfig& configInit, std::string paramName){
	QCheckBox *dataLabel = new QCheckBox(QString::fromStdString(label), this);
	dataLabel->setChecked(bool(configInit.getIProperty(GUI_CONF_PREFIX + paramName + GUI_CONF_POSTFIX_ENABLED)));
	return dataLabel;
}

QmitkAsToolGUI::QmitkAsToolGUI(){
	// Create and initialize the worker thread.
	runner = new SegRunner(m_asTool);
	runner->moveToThread(&workerThread);
	workerThread.start();

	// Set status
	activeSession = false;
	segRunning = false;

	readConfig();

	// Create the gui elements and wire them together
	buildGUI();
}

/*

Fills the configInit and configUpdates using the config file.

*/
void QmitkAsToolGUI::readConfig(){
	static bool firstInit = true;
	QmitkAsToolGUI::k++;
	if(firstInit){
		std::string progPath = mitk::IOUtil::GetProgramPath();
		char sep = mitk::IOUtil::GetDirectorySeparator();
		progPath.append(1, sep);
		std::string confFile = SETTINGS_FILENAME;
		QmitkAsToolGUI::configInit.addFromConfigFile(progPath + confFile);
		QmitkAsToolGUI::configUpdates.addFromConfigFile(progPath + confFile);
		firstInit = false;
	}
}

void QmitkAsToolGUI::buildGUI(){
	qRegisterMetaType<SimpleConfig>("SimpleConfig");
	qRegisterMetaType<p_ObjectStat>("p_ObjectStat");
	connect(this, SIGNAL(DoSegStep(SimpleConfig)), runner, SLOT(OnDoSegStep(SimpleConfig)));
	connect(this, SIGNAL(Init(mitk::Image*, mitk::Image*, SimpleConfig, int)), runner, SLOT(OnInit(mitk::Image*, mitk::Image*, SimpleConfig, int)), Qt::BlockingQueuedConnection);
	connect(this, SIGNAL(Cleanup()), runner, SLOT(OnCleanup()), Qt::BlockingQueuedConnection);
	connect(runner, SIGNAL(ReportNewSurface(p_ObjectStat)), this, SLOT(OnReportNewSurface(p_ObjectStat)));

	QGridLayout *layout = new QGridLayout(this);
	this->setContentsMargins(0, 0, 0, 0);

	// === Single cell segmentation ===

	QLabel *label = new QLabel("Single cell segmentation", this);
	QFont f = label->font();
	f.setBold(true);
	label->setFont(f);
	layout->addWidget(label, 0, 0);


	// --- Control group box --- 

	// Start segmentation

	QGroupBox *segCtrlGb = new QGroupBox("Start/stop");
	QVBoxLayout *segCtrlGbLay = new QVBoxLayout;

	startSegButton = new QPushButton(START_SEG_LABEL, this);
	connect(startSegButton, SIGNAL(clicked()), this, SLOT(OnStartButtonClicked()));
	segCtrlGbLay->addWidget(startSegButton);

	// Cancel segmentation

	stopSegButton = new QPushButton(CANCEL_SEG_LABEL, this);
	connect(stopSegButton, SIGNAL(clicked()), this, SLOT(OnStopButtonClicked()));
	stopSegButton->setEnabled(false);
	segCtrlGbLay->addWidget(stopSegButton);

	segCtrlGb->setLayout(segCtrlGbLay);
	layout->addWidget(segCtrlGb, 1, 0, 1, 2);

	// --- Auto jump between the cells --- 
	
	// Auto jump On/Off checkbox
	QCheckBox* jumpNextChck = new QCheckBox("Auto jump next object after ", this);
	autoJumpNext = false;
	
	connect(jumpNextChck, &QCheckBox::toggled, [=](bool state){
		autoJumpNext = state;
	});

	// Auto jump limit edit text
	QLabel* autoJumpLimitLabel = new QLabel("iterations.");
	autoJumpLimit = AUTO_JUMP_LIMIT_DEFAULT; // Auto jump limit default value
	
	QLineEdit *autoJumpLimitEdit = new QLineEdit(QString::number(autoJumpLimit));	
	connect(autoJumpLimitEdit, &QLineEdit::returnPressed, this, [=](){
		autoJumpLimit = atoi(autoJumpLimitEdit->text().toUtf8().constData());
	});

	QGroupBox *initGb = new QGroupBox("Segmentation initializer");
	QGridLayout *initGbLay = new QGridLayout();
	
	//QLabel *sphereRadiusLabel = new QLabel("Init sphere radius:  (0=no sphere)");
	//initGbLay->addWidget(sphereRadiusLabel, 0, 0, 1, 1);
	//QLineEdit *sphereRadiusEdit = new QLineEdit(QString::fromStdString("0"));
	//initGbLay->addWidget(sphereRadiusEdit, 0, 1, 1, 1);
	initFromSphere = new QCheckBox(QString::fromStdString("Enclose initial segmentation with a sphere?"), this);
	initGbLay->addWidget(initFromSphere, 0, 1, 1, 1);
	initGb->setLayout(initGbLay);

	segCtrlGbLay->addWidget(initGb);

	QGroupBox *autoJumpGb = new QGroupBox("Batch segmentation");
	QGridLayout *autoJumpGbLay = new QGridLayout();
	autoJumpGbLay->addWidget(jumpNextChck, 0, 0, 1, 2);
	autoJumpGbLay->addWidget(autoJumpLimitEdit, 0, 2, 1, 1);
	autoJumpGbLay->addWidget(autoJumpLimitLabel, 0, 3, 1, 2);
	autoJumpGb->setLayout(autoJumpGbLay);
	
	segCtrlGbLay->addWidget(autoJumpGb);

	// Parameters

	string cRoot = GUI_CONF_PREFIX;

	// Priors

	// volume

	QGroupBox *targetGb = new QGroupBox("Desired object properties");
	QGridLayout *targetGbLay = new QGridLayout();

	std::string targetVol = configInit[VOLUME_PRIOR_PROPERTY];
	if(savedW.count(VOLUME_PRIOR_PROPERTY) > 0){
		targetVol = std::to_string(savedW[VOLUME_PRIOR_PROPERTY]);
	}

	QLineEdit *volumeTargInp = new QLineEdit(QString::fromStdString(targetVol));
	QPushButton *manualAdjustVolBtn = new QPushButton("Adjust...");
	connect(volumeTargInp, &QLineEdit::returnPressed, this, [=](){
		float volVal = atof(volumeTargInp->text().toUtf8().constData());
		configUpdates.setFProperty(VOLUME_PRIOR_PROPERTY, atof(volumeTargInp->text().toUtf8().constData()));
		savedW[VOLUME_PRIOR_PROPERTY] = volVal;
	});

	actualVolumeVal = new QLineEdit("--");
	actualVolumeVal->setEnabled(false);
	targetGbLay->addWidget(new QLabel("Desired volume:"), 0, 0, 1, 1);
	targetGbLay->addWidget(volumeTargInp, 0, 1, 1, 1);
	targetGbLay->addWidget(new QLabel(" / "), 0, 2, 1, 1);
	targetGbLay->addWidget(actualVolumeVal, 0, 3, 1, 1);
	targetGbLay->addWidget(manualAdjustVolBtn, 0, 4, 1, 1);

	volumeTargetCSldr = new CusSlider(5);
	volumeTargetCSldr->updateIv(100, std::make_pair<float, float>(0.0f, 1.0f));
	connect(volumeTargetCSldr->getSlider(), &QSlider::sliderMoved, this, [=](int val){
		float realVal = volumeTargetCSldr->translateVal(val);
		// Set the p value and the target input value
		configUpdates.setFProperty(VOLUME_PRIOR_PROPERTY, realVal);
		volumeTargInp->setText(QString::number(realVal));
	});
	connect(manualAdjustVolBtn, &QPushButton::clicked, this, [=](){
		float volVal = atof(actualVolumeVal->text().toUtf8().constData());
		volumeTargetCSldr->updateIv(100, std::make_pair<float, float>(volVal*0.2f, volVal*1.8f));
		
		configUpdates.setFProperty(VOLUME_PRIOR_PROPERTY, volVal);
		volumeTargInp->setText(QString::number(volVal));
	});
	targetGbLay->addWidget(volumeTargetCSldr, 1, 0, 1, 5);

	// plasma

	std::string targetPlas = configInit[PLASMA_PRIOR_PROPERTY];
	if(savedW.count(PLASMA_PRIOR_PROPERTY) > 0){
		targetPlas = std::to_string(savedW[PLASMA_PRIOR_PROPERTY]);
	}

	QLineEdit *plasmaTargInp = new QLineEdit(QString::fromStdString(targetPlas));
	connect(plasmaTargInp, &QLineEdit::returnPressed, this, [=](){
		float plasVal = atof(plasmaTargInp->text().toUtf8().constData());
		configUpdates.setFProperty(PLASMA_PRIOR_PROPERTY, plasVal +  SPHERE_AMOEBA);
		savedW[PLASMA_PRIOR_PROPERTY] = plasVal;
	});

	actualPlasmaVal = new QLineEdit("--");
	actualPlasmaVal->setEnabled(false);
	QPushButton *manualAdjustPlasmaBtn = new QPushButton("Adjust...");

	targetGbLay->addWidget(new QLabel("Sphericity:"), 2, 0, 1, 1);
	targetGbLay->addWidget(plasmaTargInp, 2, 1, 1, 1);
	targetGbLay->addWidget(new QLabel(" / "), 2, 2, 1, 1);
	targetGbLay->addWidget(actualPlasmaVal, 2, 3, 1, 1);
	targetGbLay->addWidget(manualAdjustPlasmaBtn, 2, 4, 1, 1);

	plasmaTargetCSldr = new CusSlider(5);
	plasmaTargetCSldr->updateIv(100, std::make_pair<float, float>(0.0f, 1.0f));
	connect(plasmaTargetCSldr->getSlider(), &QSlider::sliderMoved, this, [=](int val){
		float realVal = plasmaTargetCSldr->translateVal(val);
		// Set the p value and the target input value
		configUpdates.setFProperty(PLASMA_PRIOR_PROPERTY, realVal + float(SPHERE_AMOEBA));
		plasmaTargInp->setText(QString::number(realVal));
	});
	connect(manualAdjustPlasmaBtn, &QPushButton::clicked, this, [=](){
		float plasVal = atof(actualPlasmaVal->text().toUtf8().constData());
		plasmaTargetCSldr->updateIv(100, std::make_pair<float, float>(plasVal*0.7f, plasVal*1.3f));

		configUpdates.setFProperty(PLASMA_PRIOR_PROPERTY, plasVal + float(SPHERE_AMOEBA));
		plasmaTargInp->setText(QString::number(plasVal));
	});

	targetGbLay->addWidget(plasmaTargetCSldr, 3, 0, 1, 5);

	targetGb->setLayout(targetGbLay);
	layout->addWidget(targetGb);

	// Weights

	QGroupBox *paramsGb = new QGroupBox("Evolution weights");
	QGridLayout *paramsGbLay = new QGridLayout();

	bool firstInit = true;
	if(firstInit){
		firstInit = false;
	}


	QCheckBox *volumeLabel = createToggleParam("Desired volume importance:", configInit, VOLUME_WEIGHT_PROPERTY);
	QDoubleSpinBox *volumeInp = createSpinBoxParam(configInit, VOLUME_WEIGHT_PROPERTY);
	volumeInp->setEnabled(volumeLabel->isChecked());
	paramsGbLay->addWidget(volumeLabel, 0, 0, 1, 1);
	paramsGbLay->addWidget(volumeInp, 0, 1, 1, 1);

	QCheckBox *plasmaLabel = createToggleParam("Desired sphericity importance:", configInit, PLASMA_WEIGHT_PROPERTY);
	QDoubleSpinBox *plasmaInp = createSpinBoxParam(configInit, PLASMA_WEIGHT_PROPERTY);
	plasmaInp->setEnabled(plasmaLabel->isChecked());
	paramsGbLay->addWidget(plasmaLabel, 1, 0, 1, 1);
	paramsGbLay->addWidget(plasmaInp, 1, 1, 1, 1);

	QCheckBox *smoothLabel = createToggleParam("Surface smoothing:", configInit, SMOOTHNESS_WEIGHT_PROPERTY);
	QDoubleSpinBox *smoothInp = createSpinBoxParam(configInit, SMOOTHNESS_WEIGHT_PROPERTY);
	smoothInp->setStyleSheet("QLabel { background-color : red; color : white; }");
	smoothInp->setEnabled(smoothLabel->isChecked());
	paramsGbLay->addWidget(smoothLabel, 2, 0, 1, 1);
	paramsGbLay->addWidget(smoothInp, 2, 1, 1, 1);

	QCheckBox *dataLabel = createToggleParam("Image importance:", configInit, DATA_WEIGHT_PROPERTY);
	QDoubleSpinBox *dataInp = createSpinBoxParam(configInit, DATA_WEIGHT_PROPERTY);
	dataInp->setEnabled(dataLabel->isChecked());
	paramsGbLay->addWidget(dataLabel, 3, 0, 1, 1);
	paramsGbLay->addWidget(dataInp, 3, 1, 1, 1);

	paramsGb->setLayout(paramsGbLay);
	layout->addWidget(paramsGb, 3, 0, 1, 2);

	/*

	Checkboxes to disable/enable the actual parameter.
	configUpdates['gui.eq.lambda.enabled'] = ...	
	
	*/
	connect(dataLabel, &QCheckBox::toggled, [=](bool state){
		dataInp->setEnabled(state);
		configUpdates[GUI_CONF_PREFIX + string(DATA_WEIGHT_PROPERTY) + GUI_CONF_POSTFIX_ENABLED] = std::to_string(int(state));
	});

	connect(smoothLabel, &QCheckBox::toggled, [=](bool state){
		smoothInp->setEnabled(state);
		configUpdates[GUI_CONF_PREFIX + string(SMOOTHNESS_WEIGHT_PROPERTY) + GUI_CONF_POSTFIX_ENABLED] = std::to_string(int(state));
	});


	connect(volumeLabel, &QCheckBox::toggled, [=](bool state){
		volumeInp->setEnabled(state);
		configUpdates[GUI_CONF_PREFIX + string(VOLUME_WEIGHT_PROPERTY) + GUI_CONF_POSTFIX_ENABLED] = std::to_string(int(state));
	});


	connect(plasmaLabel, &QCheckBox::toggled, [=](bool state){
		plasmaInp->setEnabled(state);
		configUpdates[GUI_CONF_PREFIX + string(PLASMA_WEIGHT_PROPERTY) + GUI_CONF_POSTFIX_ENABLED] = std::to_string(int(state));
	});

	// When clicked to the "Update weights", the actual inputs are savved to "savedW" and the "configUpdates" will be refreshed.
	QPushButton *updateParamsBtn = new QPushButton("Update weights");
	connect(updateParamsBtn, &QPushButton::clicked, [=](){
		double volScale = configInit.getFProperty(GUI_CONF_PREFIX + string(VOLUME_WEIGHT_PROPERTY) + GUI_CONF_POSTFIX_MUL);
		double volInpVal = atof(std::to_string(volumeInp->value()).c_str());
		double volW = volScale*volInpVal;

		double spherScale = configInit.getFProperty(GUI_CONF_PREFIX + string(PLASMA_WEIGHT_PROPERTY) + GUI_CONF_POSTFIX_MUL);
		double spherInpVal = atof(std::to_string(plasmaInp->value()).c_str());
		double spherW =  spherScale*spherInpVal;

		double smoothScale = configInit.getFProperty(GUI_CONF_PREFIX + string(SMOOTHNESS_WEIGHT_PROPERTY) + GUI_CONF_POSTFIX_MUL);
		double smoothInpVal = atof(std::to_string(smoothInp->value()).c_str());
		double smoothW =  smoothScale*smoothInpVal;

		double dataScale = configInit.getFProperty(GUI_CONF_PREFIX + string(DATA_WEIGHT_PROPERTY) + GUI_CONF_POSTFIX_MUL);
		double dataInpVal = atof(std::to_string(dataInp->value()).c_str());
		double dataW =  dataScale*dataInpVal;

		configUpdates.setFProperty(VOLUME_WEIGHT_PROPERTY, volW);
		configUpdates.setFProperty(PLASMA_WEIGHT_PROPERTY, spherW);
		configUpdates.setFProperty(SMOOTHNESS_WEIGHT_PROPERTY, smoothW);
		configUpdates.setFProperty(DATA_WEIGHT_PROPERTY, dataW);

		savedW[VOLUME_WEIGHT_PROPERTY] = atof(std::to_string(volumeInp->value()).c_str());
		savedW[PLASMA_WEIGHT_PROPERTY] = atof(std::to_string(plasmaInp->value()).c_str());
		savedW[SMOOTHNESS_WEIGHT_PROPERTY] = atof(std::to_string(smoothInp->value()).c_str());
		savedW[DATA_WEIGHT_PROPERTY] = atof(std::to_string(dataInp->value()).c_str());
	});
	layout->addWidget(updateParamsBtn, 4, 0, 1, 2);

	// Accept the result
	QGroupBox *segAccGb = new QGroupBox("Accept segmentation");
	QVBoxLayout *segAccGbLay = new QVBoxLayout;

	QPushButton *acceptSegButton = new QPushButton("Accept surface as segmentation", this);
	connect(acceptSegButton, SIGNAL(clicked()), this, SLOT(OnAcceptSegmentation()));
	segAccGbLay->addWidget(acceptSegButton);

	QPushButton *acceptSagittalSlice = new QPushButton("Accept on active slice", this);
	acceptSagittalSlice->setEnabled(false);
	connect(acceptSagittalSlice, SIGNAL(clicked()), this, SLOT(OnAcceptSagitalSlice()));
	acceptSagittalSlice->setFont(f);
	segAccGbLay->addWidget(acceptSagittalSlice);

		QGroupBox *repAddGb = new QGroupBox("Add to the current segmentation or replace?");
		QVBoxLayout *repAddGbLay = new QVBoxLayout;
		QRadioButton *addRad = new QRadioButton("Add");
		addRad->setEnabled(false);
		QRadioButton *repRad = new QRadioButton("Replace");
		repRad->setChecked(true);
		repAddGbLay->addWidget(addRad);
		repAddGbLay->addWidget(repRad);
		repAddGb->setLayout(repAddGbLay);

		segAccGbLay->addWidget(repAddGb);

	segAccGb->setLayout(segAccGbLay);
	layout->addWidget(segAccGb, 5, 0, 1, 2);

	connect(this, SIGNAL(NewToolAssociated(mitk::Tool *)), this, SLOT(OnNewToolAssociated(mitk::Tool *)));

	// === Batch segmentation ===
	QLabel *autoLabel = new QLabel("Batch mode", this);
	autoLabel->setFont(f);
	layout->addWidget(autoLabel, 6, 0, 1, 2);

	// Control the segmentation
	QGroupBox *autoSegCtrlGb = new QGroupBox("Start/stop");
	QVBoxLayout *autoSegCtrlGbLay = new QVBoxLayout;

	QPushButton* autoStartSegButton = new QPushButton("Start", this);
	connect(autoStartSegButton, SIGNAL(clicked()), this, SLOT(OnStartButtonClicked()));
	autoSegCtrlGbLay->addWidget(autoStartSegButton);

	QPushButton* autoStopSegButton = new QPushButton("Stop", this);
	connect(autoStopSegButton, SIGNAL(clicked()), this, SLOT(OnStopButtonClicked()));
	autoStopSegButton->setEnabled(false);
	autoSegCtrlGbLay->addWidget(autoStopSegButton);

	autoSegCtrlGb->setLayout(autoSegCtrlGbLay);
	//layout->addWidget(autoSegCtrlGb, 7, 0, 1, 2);


}

int QmitkAsToolGUI::getSelectedLabelId(mitk::Image* image){
	mitk::LabelSetImage *labelSetImage = dynamic_cast<mitk::LabelSetImage *>(image);
	int labelId = 1;
	if (labelSetImage){
		mitk::Label *label = labelSetImage->GetActiveLabel(labelSetImage->GetActiveLayer());
		labelId = label->GetValue();
	}
	return labelId;
}

// Segmentation control (the gui calls these methods)

// Selects the next label by updating the label id. Returns true if there are more labels, and false otherwise.
bool QmitkAsToolGUI::activateNextLabe(){

	mitk::Image::Pointer segImage = getSelectedInitializer(m_asTool->getToolManager());
	mitk::Image* pSegImage = segImage.GetPointer();
	mitk::LabelSetImage *labelSetImage = dynamic_cast<mitk::LabelSetImage *>(pSegImage);
	mitk::LabelSet* ls = labelSetImage->GetLabelSet(labelSetImage->GetActiveLayer());
	int nLabels = ls -> GetNumberOfLabels();

	if(currentWorkingLabelId < nLabels -1){
		currentWorkingLabelId++;
		return true;
	}else{
		return false;
	}
}



void QmitkAsToolGUI::launchNextStep(SimpleConfig conf){
	nIterationsElapsed++;

	if(autoJumpNext){
		if(nIterationsElapsed > autoJumpLimit){
			applyCurrentSegmentation();
			stopSegmentation();
			// Jump to the next cell
			if(activateNextLabe()){
				nIterationsElapsed = 0;
				startSegmentation();
				segRunning = true;
				launchNextStep(configUpdates);
			}			
		}else{
			emit DoSegStep(conf);	
		}
	}else{
		emit DoSegStep(conf);
	}
}

void QmitkAsToolGUI::startSegmentation(){
		// Acquire the current segmentation: we treat this as the initial one!
		mitk::DataNode::Pointer referenceDataNode = m_asTool->getToolManager()->GetReferenceData(0);
		mitk::DataNode::Pointer segmentationNode = m_asTool->getToolManager()->GetWorkingData(0);

		if(referenceDataNode.IsNull() || segmentationNode.IsNull()){
			return;
		}

		mitk::Image::Pointer refImage = dynamic_cast<mitk::Image *>(referenceDataNode->GetData());
		showDataInfo(m_asTool->getToolManager());
		mitk::Image::Pointer segImage = getSelectedInitializer(m_asTool->getToolManager());
		//mitk::Image::Pointer segImage = dynamic_cast<mitk::Image *>(segmentationNode->GetData());

		if(initFromSphere->isChecked()){
			configInit["init.strategy"] = "1";
		}else{
			configInit["init.strategy"] = "0";
		}

		emit Init(refImage.GetPointer(), segImage.GetPointer(), configInit, currentWorkingLabelId);
		nIterationsElapsed = 0;
		activeSession = true;
		stopSegButton->setEnabled(true);
}

void QmitkAsToolGUI::stopSegmentation(){
		segRunning = false;
		activeSession = false;
		startSegButton->setText(START_SEG_LABEL);
		stopSegButton->setEnabled(false);
		removeEvolutionVisualization();
		emit Cleanup();
}

//  Slots...

void QmitkAsToolGUI::OnStartButtonClicked(){

	if(!activeSession){
		currentWorkingLabelId = getSelectedLabelId(getSelectedInitializer(m_asTool->getToolManager()).GetPointer());
		startSegmentation();
	}

	if(!segRunning){
		segRunning = true;
		startSegButton->setText(PAUSE_LABEL);
		launchNextStep(configUpdates);
		//emit DoSegStep(configUpdates);
	}else{
		segRunning = false;
		startSegButton->setText(RESUME_LABEL);
	}
}

void QmitkAsToolGUI::OnStopButtonClicked(){
	if(activeSession){
		stopSegmentation();
	}
}

void QmitkAsToolGUI::OnReportNewSurface(p_ObjectStat obj){

	if(activeSession && segRunning){
		float objPlasma = pow(obj.surf, 3.0f/2.0f)/obj.vol;
		actualVolumeVal->setText(QString::fromStdString(std::to_string(obj.vol)));
		actualPlasmaVal->setText(QString::fromStdString(std::to_string(objPlasma-SPHERE_AMOEBA+1)));

		p_int3 gridSize;
		p_int3 translation;
		float* surfData = mitk::AsTool::getNextStepLevelSet(gridSize, translation);
		mitk::Surface::Pointer actualEvolutionSurface = getSurfaceFromSegmentation(surfData,
				gridSize.x, gridSize.y, gridSize.z,
				translation.x, translation.y, translation.z);
		updateOrAddEvolutionVisualization(actualEvolutionSurface);
		// TODO: Grab object stats here!
		delete surfData;
		launchNextStep(configUpdates);
		//emit DoSegStep(configUpdates);
	}
}

void QmitkAsToolGUI::OnAcceptSegmentation(){
	applyCurrentSegmentation();
}

void QmitkAsToolGUI::OnAcceptSagitalSlice(){

}

void QmitkAsToolGUI::OnNewToolAssociated(mitk::Tool *tool){
	m_asTool = dynamic_cast<mitk::AsTool*>(tool);
}

// Static methods

mitk::Image::Pointer QmitkAsToolGUI::getBinaryMaskFromSurface(mitk::Surface *surf, bool segTypeUshort, mitk::Image *refImage){
	mitk::SurfaceToImageFilter::Pointer s2iFilter = mitk::SurfaceToImageFilter::New();
	s2iFilter->MakeOutputBinaryOn();
	s2iFilter->SetUShortBinaryPixelType(segTypeUshort);
	s2iFilter->SetInput(surf);
	s2iFilter->SetImage(refImage);
	s2iFilter->Update();

	mitk::Image::Pointer newSeg = s2iFilter->GetOutput();
	return newSeg;
}

mitk::Image::Pointer QmitkAsToolGUI::getSelectedInitializer(mitk::ToolManager *toolManager){
	// List reference images:

	mitk::ToolManager::DataVectorType refImages = toolManager->GetReferenceData();

	mitk::DataNode::Pointer workingDataNode = toolManager->GetWorkingData(0);
	mitk::DataNode::Pointer referenceDataNode = toolManager->GetReferenceData(0);

	mitk::DataStorage::SetOfObjects::ConstPointer objects = toolManager->GetDataStorage()->GetDerivations(
			toolManager->GetWorkingData(0), mitk::NodePredicateDataType::New("Surface")
	);


	mitk::DataNode *selectedSurfaceNode = nullptr;
	for (mitk::DataStorage::SetOfObjects::ConstIterator it = objects->Begin(); it != objects->End(); ++it){
		mitk::DataNode::Pointer node = it.Value();
		bool selected = false;
		node->GetBoolProperty("selected", selected);
		if(node->IsSelected()){
			selectedSurfaceNode = node;
		}
	}

	// 1) If a selected surface found, then we convert it to a segmentation and use it for initialization.
	// 2) If not found, then we use the segmentation provided by the user
	// 3) Some default object (should be implemented!
	mitk::Image::Pointer initializer;
	if(selectedSurfaceNode != nullptr){
		initializer = getBinaryMaskFromSurface(
				dynamic_cast<mitk::Surface *>(selectedSurfaceNode->GetData()),
				true,
				dynamic_cast<mitk::Image *>(referenceDataNode->GetData()));
	}else{
		initializer = dynamic_cast<mitk::Image *>(workingDataNode->GetData());
	}

	return initializer;
}

void QmitkAsToolGUI::showDataInfo(mitk::ToolManager *toolManager){
	mitk::DataNode::Pointer workingDataNode = toolManager->GetWorkingData(0);
	mitk::DataNode::Pointer referenceDataNode = toolManager->GetReferenceData(0);

	//mitk::DataStorage::SetOfObjects::ConstPointer objects = toolManager->GetDataStorage()->GetAll();
	mitk::ToolManager::DataVectorType objects = toolManager->GetWorkingData();

	for(unsigned int i = 0; i < objects.size(); i++){
		mitk::DataNode::Pointer node = objects[i];
	}

}

// Other methods ...

// Called upon the stop segmentation...
void QmitkAsToolGUI::removeEvolutionVisualization(){
	m_asTool->getToolManager()->GetDataStorage()->Remove(m_surfaceDataNode);
}

mitk::Surface::Pointer QmitkAsToolGUI::getSurfaceFromSegmentation(float* surfData,
		int w, int h, int d,
		int tx, int ty, int tz){

	// Convert the c-style float arrray to a vtkImageData
	vtkSmartPointer<vtkImageImport> imageImport =
	vtkSmartPointer<vtkImageImport>::New();
	imageImport->SetDataSpacing(1, 1, 1);
	imageImport->SetDataOrigin(0, 0, 0);
	imageImport->SetWholeExtent(0, w-1, 0, h-1, d-1, 0);
	imageImport->SetDataExtentToWholeExtent();
	imageImport->SetDataScalarTypeToFloat();
	imageImport->SetNumberOfScalarComponents(1);
	imageImport->SetImportVoidPointer(surfData);
	imageImport->Update();


	// vtkImage -> vtkPolyData
	vtkSmartPointer<vtkMarchingCubes> marchingCubes = vtkSmartPointer<vtkMarchingCubes>::New();

	vtkImageData *o = imageImport->GetOutput();
	o->SetDimensions(w, h, d);
	marchingCubes->SetInputData(o);

	marchingCubes->ComputeNormalsOn();
	float isoValue = 0.0f;
	marchingCubes->SetValue(0, isoValue);
	marchingCubes->Update();

	// Transform: vtkPolyData->vtkPolyData

	vtkSmartPointer<vtkTransform> translation = vtkSmartPointer<vtkTransform>::New();
	translation->Translate(-float(tx), -float(ty), -float(tz));

	vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	transformFilter->SetInputConnection(marchingCubes->GetOutputPort());
	transformFilter->SetTransform(translation);
	transformFilter->Update();

	mitk::Surface::Pointer segmentationResult = mitk::Surface::New();
	segmentationResult->SetVtkPolyData(transformFilter->GetOutput(), 0);
	return segmentationResult;
}

void replaceImageContents(mitk::Image* src, mitk::Image* target, unsigned int timestep){
	mitk::ImageReadAccessor readAccess(src, src->GetVolumeData(timestep));
	const void *cPointer = readAccess.GetData();

	if (target && cPointer){
		target->SetVolume(cPointer, timestep, 0);
	}else{
		// TODO: throw exception!
		return;
	}

}

void addSegmentation(mitk::Image* src, mitk::Image* target, unsigned int timestep, int newSegLabel){
	mitk::ImageReadAccessor readAccessSrc(src, src->GetVolumeData(timestep));
	const uint16_t *cPointer = (const uint16_t*) readAccessSrc.GetData();

	mitk::ImageWriteAccessor writeAccessTarget(target, target->GetVolumeData(timestep));
	uint16_t *vPointer = (uint16_t*) writeAccessTarget.GetData();

	int vol = src->GetDimension(0) * src->GetDimension(1) * src->GetDimension(2);
	for(int i = 0; i < vol; i++){
		uint16_t colSrc = cPointer[i];
		if(colSrc == 1){
			vPointer[i] = newSegLabel;
		}
	}
}

void replaceSegmentation(mitk::Image* src, mitk::Image* target, unsigned int timestep, int newSegLabel){
	mitk::ImageReadAccessor readAccessSrc(src, src->GetVolumeData(timestep));
	const uint16_t *cPointer = (const uint16_t*) readAccessSrc.GetData();

	mitk::ImageWriteAccessor writeAccessTarget(target, target->GetVolumeData(timestep));
	uint16_t *vPointer = (uint16_t*) writeAccessTarget.GetData();

	int vol = src->GetDimension(0) * src->GetDimension(1) * src->GetDimension(2);
	for(int i = 0; i < vol; i++){
		uint16_t colTarget = vPointer[i];
		if(colTarget == newSegLabel){
			vPointer[i] = 0;
		}
		
		uint16_t colSrc = cPointer[i];
		if(colSrc == 1){
			vPointer[i] = newSegLabel;
		}
	}
}


void QmitkAsToolGUI::initEvolutionVisualizationSurface(){
	// Create the data node for the object if does not exist!
	if(m_surfaceDataNode.IsNull()){
		m_surfaceDataNode = mitk::DataNode::New();
		m_surfaceDataNode -> SetName("Contour evolution status");
		float SURFACE_COLOR_RGB[3] = {0.49f, 1.0f, 0.16f};
		m_surfaceDataNode->SetProperty("color", mitk::ColorProperty::New(SURFACE_COLOR_RGB));
		m_surfaceDataNode->SetProperty("name", mitk::StringProperty::New("Active surfaces segmentation result."));
		m_surfaceDataNode->SetProperty("opacity", mitk::FloatProperty::New(0.5));
		m_surfaceDataNode->SetProperty("line width", mitk::FloatProperty::New(4.0f));
		m_surfaceDataNode->SetProperty("includeInBoundingBox", mitk::BoolProperty::New(false));
		m_surfaceDataNode->SetProperty("helper object", mitk::BoolProperty::New(true));
		m_surfaceDataNode->SetVisibility(false);
	}
}

/*
 * Replaces the data in the active surfaces evolution step visualisation data node (m_surfaceDataNode), and re-renders it.
 * If it is not exits yet, then the function creates a new one.
 */
void QmitkAsToolGUI::updateOrAddEvolutionVisualization(mitk::Surface::Pointer a_surface){
	initEvolutionVisualizationSurface();

	mitk::DataStorage *dataStorage = m_asTool->getToolManager()->GetDataStorage();

	if(!dataStorage){
		std::cout << "__ mitkAsTool: No data storage has found!" << std::endl;
		return;
	}

	mitk::DataNode *workingNode = m_asTool->getToolManager()->GetWorkingData(0);

	if(!workingNode){
		std::cout << "__ mitkAsTool: No working node!" << std::endl;
	}

	// Create the object and put it into a data node!

	mitk::Surface::Pointer segmentationResult = a_surface; //getSegSurface();

	m_surfaceDataNode->SetData(segmentationResult);

	// Add the data node to the storage if it is not in.
	if (!dataStorage->Exists(m_surfaceDataNode)){
		std::cout << "Surface node does not exist!" << std::endl;
		dataStorage->Add(m_surfaceDataNode);
	}

	m_surfaceDataNode->SetVisibility(true);
	mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}

void QmitkAsToolGUI::applyCurrentSegmentation(){
	if (m_surfaceDataNode.IsNotNull() && m_surfaceDataNode->GetData()){


		mitk::DataNode* workData_0 = m_asTool->getToolManager()->GetWorkingData(0);
		mitk::DataNode* refData_0 = m_asTool->getToolManager()->GetReferenceData(0);

		if (refData_0 == nullptr || workData_0 == nullptr){
		  return;
		}

		mitk::DataNode *segmentationNode = workData_0;
		mitk::Image *currSeg = dynamic_cast<mitk::Image *>(segmentationNode->GetData());

		mitk::Image::Pointer newSeg = getBinaryMaskFromSurface(
				dynamic_cast<mitk::Surface *>(m_surfaceDataNode->GetData()),
				currSeg->GetPixelType().GetComponentType() == itk::ImageIOBase::USHORT,
				dynamic_cast<mitk::Image *>(refData_0->GetData()));

		//unsigned int timestep = m_LastSNC->GetTime()->GetPos();
		unsigned int timestep = 0;
		//replaceImageContents(newSeg, currSeg, timestep, newlabelId);
		//addSegmentation(newSeg, currSeg, timestep, newLabelId);
		replaceSegmentation(newSeg, currSeg, timestep, currentWorkingLabelId);
		//m_CmbInterpolation->setCurrentIndex(0);
		mitk::RenderingManager::GetInstance()->RequestUpdateAll();
		mitk::DataNode::Pointer segSurface = mitk::DataNode::New();
		float rgb[3];
		segmentationNode->GetColor(rgb);
		segSurface->SetColor(rgb);
		segSurface->SetData(m_surfaceDataNode->GetData());
		std::stringstream stream;
		stream << segmentationNode->GetName();
		stream << "_";
		stream << "Active Surfaces suggestion";
		segSurface->SetName(stream.str());
		segSurface->SetProperty("opacity", mitk::FloatProperty::New(0.7));
		segSurface->SetProperty("includeInBoundingBox", mitk::BoolProperty::New(true));
		segSurface->SetVisibility(false);
		m_asTool->getToolManager()->GetDataStorage()->Add(segSurface, segmentationNode);
		segSurface->SetVisibility(true);
		m_surfaceDataNode->SetVisibility(false);
	}
}

QmitkAsToolGUI::~QmitkAsToolGUI(){
	if(activeSession){
		stopSegmentation();
	}
	workerThread.quit();
	workerThread.wait();
	delete runner;
}

/**
 * SegRunner class
 */

void SegRunner::OnInit(mitk::Image* im, mitk::Image* seg, SimpleConfig conf, int labelId){
	m_asTool->init(im, seg, conf, labelId);
}

void SegRunner::OnDoSegStep(SimpleConfig conf){
	p_ObjectStat stat = m_asTool->launchNextStepComputation(conf);
	emit ReportNewSurface(stat);

}

void SegRunner::OnCleanup(){
	m_asTool->cleanup();
}
