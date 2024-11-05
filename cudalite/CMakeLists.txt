cmake_minimum_required(VERSION 3.30)
project(libcudalite CUDA CXX)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr")

include_directories(
    "../cutlass/include"
    "../cutlass/tools/util/include")

add_library(libzen SHARED static/cudalite.def
    gemm.cu
    conv.cu)

target_include_directories(libzen PRIVATE "./include")

set_target_properties(libzen PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        LINKER_LANGUAGE CXX)

target_link_libraries(libzen PRIVATE CUDA::cudart_static)
