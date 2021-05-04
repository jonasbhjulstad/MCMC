cmake_minimum_required(VERSION 3.0.0)
project(Custom_MCMC VERSION 0.1.0)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
set(CMAKE_CXX_STANDARD_REQUIRED YES)   # See below (1)
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_BUILD_TYPE Debug)

add_executable(${PROJECT_NAME} main.cc)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

add_subdirectory(include)
add_subdirectory(MCMC_Sampler)
add_subdirectory(SIR)
add_subdirectory(MCMC_Model)

target_include_directories(${PROJECT_NAME} PUBLIC MCMC_Sampler MCMC_Model include SIR)

FIND_PACKAGE( Boost 1.40 COMPONENTS program_options REQUIRED )
find_package(Boost REQUIRED COMPONENTS thread)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
find_package(GSL REQUIRED)    # See below (2)

TARGET_LINK_LIBRARIES( ${PROJECT_NAME} LINK_PUBLIC ${Boost_LIBRARIES} GSL::gsl GSL::gslcblas libsmctc.a SIR Boost::thread pthread)


