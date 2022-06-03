file(GLOB MCMC_SOURCES "*.cpp")
add_library(MCMC STATIC ${MCMC_SOURCES})
target_include_directories(MCMC PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})



