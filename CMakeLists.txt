cmake_minimum_required(VERSION 3.0.0)
project(CLCPP_MCMC)
include(FetchContent)



set(CMAKE_BUILD_TYPE Debug)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY Executables)

find_package(OpenCL REQUIRED)
find_package(OpenCLHeaders REQUIRED)
if(UNIX)
set(SPIRV_COMPILER "/usr/local/bin/clang-14")
set(CLCPP_MCMC_LLVM_SPIRV "/usr/local/bin/llvm-spirv")
find_package(OpenCLICDLoader REQUIRED)
endif()
if(WIN32)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED 17)
set(CMAKE_CXX_STANDARD 17)
find_package(CLCPP_PRNG REQUIRED HINTS "/home/deb/Documents/CLCPP_PRNG/build/pkg/")
message(WARNING ${CLCPP_PRNG_INCLUDE_DIR})
if (UNIX)
find_package(CLG_Utils REQUIRED HINTS "/home/deb/Documents/OpenCL-Graph/build/pkg/")
endif()

set(CLCPP_MCMC_KERNEL_SOURCE_DIR "${CMAKE_SOURCE_DIR}/Kernels/")
set(CLCPP_MCMC_KERNEL_DECOMPOSITIONS_SOURCE_DIR "${CLCPP_MCMC_KERNEL_SOURCE_DIR}/Decompositions/")
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/data)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/data/SIR_Stochastic)
set(CLCPP_MCMC_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/data")
add_subdirectory(Utils)
add_subdirectory(Kernels)
add_subdirectory(include)
add_subdirectory(Generate)
add_subdirectory(Simulation_Scripts)
include_directories(PUBLIC include)
include_directories(PUBLIC Simulation_Scripts)



