cmake_minimum_required(VERSION 3.0)
project(Patch)

set(PATCH_DIR "mitk-patch" CACHE PATH "Select the patch dir!")
set(MITK_PATH CACHE PATH "Select the MITK project path!")

configure_file(patch.py.in patch.py @ONLY)

set(FILE_LIST "filelist.txt")

add_custom_command(OUTPUT ${FILE_LIST}
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/${FILE_LIST} "${CMAKE_CURRENT_BINARY_DIR}/${FILE_LIST}")

