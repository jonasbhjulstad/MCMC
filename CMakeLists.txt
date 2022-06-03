cmake_minimum_required(VERSION 3.0.0)
project(OpenCL_MCMC)
include(FetchContent)



set(CMAKE_CXX_STANDARD_REQUIRED YES)   # See below (1)
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_BUILD_TYPE Debug)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY Executables)

set(CMAKE_CXX_STANDARD_REQUIRED 17)
set(CMAKE_CXX_STANDARD 17)
FetchContent_Declare(randomCL_repo
GIT_REPOSITORY https://github.com/bstatcomp/RandomCL.git
)
FetchContent_MakeAvailable(randomCL_repo)


add_subdirectory(include)
add_subdirectory(Generate)
add_subdirectory(Simulation_Scripts)
include_directories(PUBLIC include)
include_directories(PUBLIC Simulation_Scripts)



