cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if (WIN32)
	set( BOOST_ROOT "C:/local/boost_1_69_0" CACHE FILEPATH "Boost install root directory.")
	# Bypass the CUDA compiler check because of a bug on Windows...
	set(CMAKE_CUDA_COMPILER_WORKS TRUE)
endif (WIN32)

project(Seg3dProject LANGUAGES CXX CUDA)

find_package( OpenCV COMPONENTS core imgcodecs REQUIRED )
find_package( Boost 1.65 COMPONENTS system filesystem log timer chrono REQUIRED )

# Disable lib inference on Win
add_definitions( "-DBOOST_ALL_NO_LIB" )

set(SRCS 
	src/SimpleConfig.cu
	src/buildGraph.cu
	src/imagetools.cu
	src/surfaceAndVolume.cu
	src/common.cu
	src/matrix.cu
	src/narrowBand.cu
	src/evolve.cu
	src/phaseField.cu
	src/marchingCubes.cu
	src/segmentation.cu
	src/localRegion.cu

	# ccalg submodule
	src/ccalg/computeComponents.cu

	# cudatools submodule
	src/cudatools/cudaErr.cu
	src/cudatools/errorHandling.cu
	src/cudatools/function.cu  
	src/cudatools/types.cu
)


set( MAIN src/main.cu)

if (WIN32)
	#To fix the same bug on Windows.
	set( CUDA_LIB "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/lib/x64" CACHE FILEPATH "One should set up the lib/x64 subdirectory of the CUDA toolkit sometimes on Windows.")
	link_directories( ${CUDA_LIB} )
endif (WIN32)

#set( APPNAME seg3dExec )
set( STATNAME slective_static )
set( SHNAME selective_shared )

include_directories( ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} )

# Create a static library from the soruces
#add_definitions( -D_DC_OP_LOG_CUDA=1 )
add_library( ${STATNAME} STATIC ${SRCS})
set_target_properties( ${STATNAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
target_compile_definitions( ${STATNAME} PUBLIC  BOOST_LOG_DYN_LINK )

# Create executable using the main.cu, link with the static lib
# add_executable( ${APPNAME} ${MAIN} )
# set_property( TARGET ${APPNAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON )
# target_link_libraries( ${APPNAME} PRIVATE ${STATNAME} ${Boost_LIBRARIES} ${OpenCV_LIBS} cuda)

# Create shared library
set( SRCS_LIB ${SRCS} ${MAIN})
add_library( ${SHNAME} SHARED ${SRCS_LIB})
set_target_properties( ${SHNAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
target_compile_definitions( ${SHNAME} PUBLIC BOOST_LOG_DYN_LINK )
set_property( TARGET ${SHNAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON )

target_link_libraries( ${SHNAME} PUBLIC ${Boost_LIBRARIES} ${OpenCV_LIBS} cuda)

add_custom_target(print ${CMAKE_COMMAND} -E echo $<TARGET_FILE_DIR:selective_shared>)

#install( TARGETS ${APPNAME} 
#		 CONFIGURATIONS Release
#		 RUNTIME DESTINATION Release_/bin
#		 LIBRARY DESTINATION lib
#)

string(APPEND CMAKE_CUDA_FLAGS " -lineinfo")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_30,code=compute_30")

if ( APPLE )
	set_property( TARGET 3d-segmentation PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} )
	set_property( TARGET test-generator PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} )
endif()

# Copy the include files needed by other projects

set(INC_DIR "include")

add_custom_command(
	TARGET ${SHNAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/${INC_DIR}")

set(COPY_FILES "selective.h" "SimpleConfig.cuh" "macros.h")

foreach( f IN ITEMS ${COPY_FILES} )
	add_custom_command(
		TARGET ${SHNAME} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/src/${f}" "${CMAKE_CURRENT_BINARY_DIR}/${INC_DIR}/${f}")
endforeach( f )

# Copy the settings

set(COPY_CONF_FILES "settings_3DCA.conf")

foreach( f IN ITEMS ${COPY_CONF_FILES} )
	add_custom_command(
		TARGET ${SHNAME} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/${f}" "${CMAKE_CURRENT_BINARY_DIR}/${f}")
endforeach( f )

# Copy the binary cmake project file

add_custom_command(
	TARGET ${SHNAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/SelectiveConfig.cmake "${CMAKE_CURRENT_BINARY_DIR}/SelectiveConfig.cmake")
