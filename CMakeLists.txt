cmake_minimum_required(VERSION 3.0.0)
project(SIR_EX VERSION 0.1.0)
project(SEIR_EX VERSION 0.1.0)
project(SEIAR_EX VERSION 0.1.0)
project(Gen_SIR VERSION 0.1.0)
project(Gen_SEIR VERSION 0.1.0)
project(Gen_SEIAR VERSION 0.1.0)

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

set(SIR_SOURCES Simulation_Scripts/run_SIR.cc src/SIR_Stochastic.cc src/SMC_Sampler.cc src/MCMC_Sampler.cc)
set(SEIR_SOURCES Simulation_Scripts/run_SEIR.cc src/SEIR_Stochastic.cc src/SMC_Sampler.cc src/MCMC_Sampler.cc)
set(SEIAR_SOURCES Simulation_Scripts/run_SEIAR.cc src/SEIAR_Stochastic.cc src/SMC_Sampler.cc src/MCMC_Sampler.cc)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY Executables)

add_executable(SIR_EX ${SIR_SOURCES})
add_executable(SEIR_EX ${SEIR_SOURCES})
add_executable(SEIAR_EX ${SEIAR_SOURCES})
add_executable(Gen_SIR Generate/Generate_SIR_data.cc)
add_executable(Gen_SEIR Generate/Generate_SEIR_data.cc)
add_executable(Gen_SEIAR Generate/Generate_SEIAR_data.cc)


add_subdirectory(include)
add_subdirectory(src)
include_directories(PUBLIC include)
include_directories(PUBLIC Simulation_Scripts)

# FIND_PACKAGE( Boost 1.40 COMPONENTS program_options REQUIRED )
# find_package(Boost REQUIRED COMPONENTS thread)
# INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
find_package(GSL REQUIRED)    # See below (2)

TARGET_LINK_LIBRARIES(SIR_EX LINK_PUBLIC GSL::gsl GSL::gslcblas)
TARGET_LINK_LIBRARIES(SEIR_EX LINK_PUBLIC GSL::gsl GSL::gslcblas)
TARGET_LINK_LIBRARIES(SEIAR_EX LINK_PUBLIC GSL::gsl GSL::gslcblas)


