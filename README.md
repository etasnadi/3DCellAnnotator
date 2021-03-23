3D Cell Annotator software.

It consists of a CUDA implementation of the selective segmentation method (SelectiveSegmentation) and source code patch to the segmentation module of the MITK to interact with the CUDA implementation.

Build instructions:

1. Clone and checkout the 2018.04 version of MITK: ```https://github.com/MITK/MITK/tree/v2018.04``` (commit ac93ed9baf)
	
	Make sure it can be built without errors using ```cmake``` and ```make```.

2. Clone the 3DCellAnnotator source: ```git clone https://github.com/etasnadi/3DCellAnnotator.git```

	1. ```cmake``` and ```make``` the ```SelectiveSegmentation``` project.

	2. ```cmake``` the ```MITKIntegration``` project. (Set the ```MITK_PATH``` variable to the root of the MITK source and ```PATCH_DIR``` to the ```MITKIntegration/mitk-patch``` directory in the 3DCellAnnotator source tree).
	In the build dir execute ```python patch.py apply``` (this project only intended to generate a python script that copies/updates the required code for the MITK interface of the ```SelectiveSegmentation project```.

3. Go to the ```MITK-build``` directory under the MITK superbuild, and:

	Execute the```cmake```command again and set the ```Selective_DIR``` to the build directory of the ```SelectiveSegmentation``` using the ```cmake-gui``` or something on your system.
	Also, enable the ```MITK_BUILD_org.mitk.gui.qt.segmentation``` and ```MITK_BUILD_org.mitk.gui.qt.multilabelsegmentation``` checboxes to include the MITK's segmentation plugin in the build.
	```make install```.

	Copy the configuration file ```settings_3DCA.conf``` from the build dir of the ```SelectiveSegmentation``` project to the installed porject's ```bin``` directory.

The app can be executed using ```bin/MitkWorkbench```.

See issue [#2] for a detailed example of how to build on Ubuntu 20.04!
