cmake_minimum_required(VERSION 3.30)
project(cudalite CUDA CXX)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 20)

add_library(libcudalite SHARED cudalite.def)

set_target_properties(libcudalite PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        LINKER_LANGUAGE CXX)

target_link_libraries(libcudalite PRIVATE CUDA::cudart_static)
