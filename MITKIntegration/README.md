Integrates the selectve segmentation CUDA implementation into the MITK segmentation module by patching the source code.

```cmake . && cmake-gui .``` and set up the ``` MITK_PATH``` and  ```PATCH_DIR``` on the gui (the first one is the MITK source dir, the second one is the path where the modifications are collected, a subdir in this repo) and ```python3 patch.py create``` (creates a patch to the ```PATCH_DIR``` dir while ```python3 patch.py apply``` applies the patch from the ```PATCH_DIR``` to ```MITK_PATH```.
