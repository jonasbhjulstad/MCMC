cmake_minimum_required(VERSION 3.0.0)
project(SYCL_MCMC)
find_package(IntelDPCPP REQUIRED)
find_package(MKL REQUIRED HINTS "/home/arch/intel/oneapi/mkl/2022.1.0/lib/cmake/")
find_package(oneDPL REQUIRED HINTS "/home/arch/intel/oneapi/dpl/latest/lib/cmake/oneDPL")
find_package(SUNDIALS REQUIRED)
include(FetchContent)
enable_testing()
FetchContent_Declare(
    ranluxpp_repo
    GIT_REPOSITORY https://github.com/jirka-h/ranluxpp-portable.git
    GIT_TAG origin/main
)
if (NOT ranluxpp_repo_POPULATED)
FetchContent_Populate(ranluxpp_repo)
endif()

file(GLOB ranluxpp_SOURCE ${ranluxpp_repo_SOURCE_DIR}/*.c)
add_library(ranluxpp STATIC ${ranluxpp_SOURCE})
target_include_directories(ranluxpp PUBLIC ${ranluxpp_repo_SOURCE_DIR})

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


set(CMAKE_CXX_STANDARD_REQUIRED 17)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(Kernels)
add_subdirectory(include)
add_subdirectory(Generate)
add_subdirectory(test)
add_subdirectory(Executables)


