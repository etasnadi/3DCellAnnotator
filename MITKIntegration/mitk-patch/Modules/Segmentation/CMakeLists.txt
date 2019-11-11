find_package( Selective REQUIRED )

MITK_CREATE_MODULE(
  INCLUDE_DIRS Algorithms Controllers DataManagement Interactions Rendering SegmentationUtilities/BooleanOperations SegmentationUtilities/MorphologicalOperations ${SELECTIVE_INCLUDE_DIRS}
  DEPENDS MitkAlgorithmsExt MitkIpSegmentation MitkIpFunc MitkSurfaceInterpolation MitkGraphAlgorithms MitkContourModel MitkMultilabel
  PACKAGE_DEPENDS
  PUBLIC ITK|ITKBinaryMathematicalMorphology+ITKLabelVoting+ITKRegionGrowing+ITKFastMarching+ITKAnisotropicSmoothing+ITKWatersheds)

include_directories(${SELECTIVE_INCLUDE_DIRS})
target_link_libraries(${MODULE_TARGET} PUBLIC ${SELECTIVE_SHARED})

add_subdirectory(Testing)
