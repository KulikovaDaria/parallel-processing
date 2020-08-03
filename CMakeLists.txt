cmake_minimum_required(VERSION 3.6)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")

project(OpenMP)

include_directories(.)

add_subdirectory(solving-ordinary-differential-equations)
add_subdirectory(solving-polynomial-equations)
add_subdirectory(vector-and-matrix-operations)