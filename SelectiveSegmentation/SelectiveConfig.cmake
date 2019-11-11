cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(Selective)

set(SELECTIVE_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include")

if (UNIX)
	set(SELECTIVE_SHARED "${CMAKE_CURRENT_LIST_DIR}/libselective_shared.so")
endif ()

if (WIN32)
	set(SELECTIVE_SHARED "${CMAKE_CURRENT_LIST_DIR}/Debug/selective_shared.lib")
endif (WIN32)
