3D Cell Annotator software.

It consists of a CUDA implementation of the selective segmentation method (SelectiveSegmentation) and source code patch to the segmentation module of the MITK to interact with the CUDA implementation.

Build instructions:

1. Download MITK 2018.04 src: http://mitk.org/wiki/Downloads
Optional: make sure it can be built without errors:
```make```

2. Clone 3DCellAnnotator source
```git clone https://github.com/etasnadi/3DCellAnnotator.git```

	1. ```cmake``` and ```make``` the ```SelectiveSegmentation``` project.

	2. ```cmake``` the ```MITKIntegration``` project. (set the ```MITK_PATH``` to the root of the MITK source and ```PATCH_DIR``` to the ```MITKIntegration``` directory in the source tree)
	In the build dir: ```python3 patch.py apply``` (this will copies the required files into the mitk source tree)

3. Go to the ```MITK-build``` directory in the mitk build dir: 
Set the ```cmake``` and set the ```Selective_DIR``` to the build directory of the ```SelectiveSegmentation``` using the ```cmake-gui``` or something.
Also, enable the ```MITK_BUILD_org.mitk.gui.qt.segmentation``` and ```MITK_BUILD_org.mitk.gui.qt.multilabelsegmentation``` checboxes to include them into the build
```make install```.

Copy the configuration file ```settings_3DCA.conf``` from the build dir of the ```SelectiveSegmentation``` to the installed porjects ```bin``` directory.
