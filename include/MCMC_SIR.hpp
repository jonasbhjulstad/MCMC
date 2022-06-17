#ifndef CLCPP_MCMC_SIR_HPP
#define CLCPP_MCMC_SIR_HPP
#include <string>
#include <stdexcept>
#include <CLG.hpp>
#include <CLCPP_MCMC_FilePaths.hpp>
#include <random>
#include <csv_reader.hpp>
#include <limits>
#include <utility>
#define SPIRV_COMPILER "/usr/local/bin/clang-14"

class MCMC_SIR_Kernel
{
    // std::string kernel_file = std::string(CLCPP_MCMC_KERNEL_SOURCE_DIR) + "/MCMC_Epidemiological";
    const size_t N_param = 2;
    const size_t  Nx = 3;

    int status = 0;
    cl_mem seedBuffer;
    cl_mem x0_Buffer;
    cl_mem yBuffer;
    cl_mem initParamBuffer;
    cl_mem paramBuffer;
    cl_mem prop_std_Buffer;
    cl_mem logSumWeightBuffer;
    cl_kernel kernel;
    std::vector<float> y;
    float dt;
    size_t N_observations, N_MCMC, N_particles, N_workers, paramBufferLength;
    std::string kernel_file = std::string(CLCPP_MCMC_KERNEL_SOURCE_DIR) + "testKernel";
    const std::string spirv_emit_command = std::string(SPIRV_COMPILER) + " --target=spirv32 -c -cl-kernel-arg-info -cl-std=clc++2021 -emit-llvm " + kernel_file + ".clcpp -o " + kernel_file + ".ll";
    const std::string spirv_compile_command = std::string(SPIRV_COMPILER) + " -Wall -c --target=spirv32 -cl-std=clc++2021 " + kernel_file + ".clcpp -o " + kernel_file + ".spv";
    std::string warning_flags = " -Wimplicit-const-int-float-conversion ";

    std::string preprocessor_definitions;

    const std::string kernel_include_directories = " -I " + std::string(CLCPP_MCMC_KERNEL_DECOMPOSITIONS_DIR) + 
    " " + std::string(CLCPP_PRNG_INCLUDE) + 
    " -I " + std::string(CLG_KERNEL_DIR) + "/Epidemiological/";

    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
    std::uniform_int_distribution<cl_ulong> dist {0, std::numeric_limits<cl_ulong>::max()};
    std::vector<cl_ulong> seeds;
    CLG_Instance& clInstance;
public:
// extern void load_data(std::string, double**, double*, const long &);
MCMC_SIR_Kernel(CLG_Instance& _clInstance, const std::string& filePath, size_t _N_MCMC, size_t _N_particles, size_t _N_workers, CLG_PRNG_TYPE prng_type): clInstance(_clInstance), N_MCMC(_N_MCMC), N_particles(_N_particles), N_workers(_N_workers)
{
    float** p_data;
    N_observations = load_data(filePath, p_data, &dt, Nx);
    y.resize(N_observations);
    for (int i = 0; i < N_observations; i++)
    {
        y[i] = p_data[i][1];
    }
    seeds.resize(N_workers);

    preprocessor_definitions = " -D PRNG_GENERATOR=" + CLG_PRNG_class_strmap.at(prng_type) + 
     + " -D N_OBSERVATIONS=" + std::to_string(N_observations)
     + " -D N_MCMC_ITERATIONS=" + std::to_string(_N_MCMC)
     + " -D N_PARTICLES=" + std::to_string(N_particles);
    paramBufferLength = N_param*(N_MCMC+1);

}

// int compile()
// {
    // return std::system((spirv_compile_command + warning_flags + preprocessor_definitions + kernel_include_directories).c_str());
// }

void compile()
{
    std::system((spirv_emit_command + preprocessor_definitions + kernel_include_directories + warning_flags).c_str()); 
    std::system((std::string(CLCPP_MCMC_LLVM_SPIRV) + " " + std::string(kernel_file) + ".ll -o " + std::string(kernel_file) + ".spv").c_str());
}


void build()
{
    std::string kernel_file = std::string(CLCPP_MCMC_KERNEL_SOURCE_DIR) + "testKernel";

    int err = 0;
    std::string programBinary = convertToString((kernel_file + ".spv").c_str());
    long unsigned int programSize = sizeof(char)*programBinary.length();

    clInstance.program = clCreateProgramWithIL(clInstance.context, (const void*) programBinary.data(), sizeof(char)*programBinary.length(), &err);
    assert(err == CL_SUCCESS);

    std::string build_options = "";
    /*Step 6: Build program. */
    int status = clBuildProgram(clInstance.program, 1, clInstance.device_ids.data(), build_options.c_str(), NULL, NULL);
    assert(status == CL_SUCCESS);
    CLG_print_build_log(status, clInstance);
}

void createBuffers()
{
    cl_int err;
    size_t seedBufferSize = N_workers * sizeof(cl_ulong);

    seedBuffer = clCreateBuffer(clInstance.context, CL_MEM_READ_ONLY,
                                       seedBufferSize, NULL, &err);
    assert(err == CL_SUCCESS);

    x0_Buffer = clCreateBuffer(clInstance.context, CL_MEM_READ_ONLY,
                                     sizeof(float)*Nx, NULL, &err);
    assert(err == CL_SUCCESS);
    
    yBuffer = clCreateBuffer(clInstance.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float)*N_observations, (void *) y.data(), &err);
    assert(err == CL_SUCCESS);
    
    initParamBuffer = clCreateBuffer(clInstance.context, CL_MEM_READ_ONLY,
                                            sizeof(float)*N_param, NULL, &err);
    assert(err == CL_SUCCESS);


    paramBuffer = clCreateBuffer(clInstance.context, CL_MEM_WRITE_ONLY,
                                         paramBufferLength* sizeof(float), NULL, &err);
    assert(err == CL_SUCCESS);
    
    prop_std_Buffer = clCreateBuffer(clInstance.context, CL_MEM_READ_ONLY,
                                            sizeof(float)*N_param, NULL, &err);
    assert(err == CL_SUCCESS);
    
    logSumWeightBuffer = clCreateBuffer(clInstance.context, CL_MEM_WRITE_ONLY,
                                                (N_MCMC+1)*sizeof(float), NULL, &err);
    assert(err == CL_SUCCESS);
}

void assignParameters(const float* x0, const float* initParam, const float* prop_std)
{
    cl_int err;
    err = clEnqueueWriteBuffer(clInstance.commandQueue, x0_Buffer, CL_TRUE, 0, sizeof(float)*Nx, x0, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
    err = clEnqueueWriteBuffer(clInstance.commandQueue, initParamBuffer, CL_TRUE, 0, sizeof(float)*N_param, initParam, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
    err = clEnqueueWriteBuffer(clInstance.commandQueue, prop_std_Buffer, CL_TRUE, 0, sizeof(float)*N_param, prop_std, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}

void createKernel()
{
    cl_int err;
    kernel = clCreateKernel(clInstance.program, "testKernel", &err);
    assert(err == CL_SUCCESS);
}

void setKernelArgs(size_t N_pop, float ll_std, float nu_I, float nu_R, float resampleThreshold)
{
    cl_int err;
    /*Step 8: Create kernel object */
    /*Step 9: Sets Kernel arguments.*/
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&seedBuffer);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&x0_Buffer);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&yBuffer);
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&initParamBuffer);
    status = clSetKernelArg(kernel, 4, sizeof(float), (void *)&dt);
    status = clSetKernelArg(kernel, 5, sizeof(float), (void *)&N_pop);
    status = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&prop_std_Buffer);
    status = clSetKernelArg(kernel, 7, sizeof(float), (void *)&ll_std);
    status = clSetKernelArg(kernel, 8, sizeof(float), (void*)&nu_I);
    status = clSetKernelArg(kernel, 9, sizeof(float), (void*)&nu_R);
    status = clSetKernelArg(kernel, 10, sizeof(float), (void*)&resampleThreshold);
    status = clSetKernelArg(kernel, 11, sizeof(cl_mem), (void *)&paramBuffer);
    status = clSetKernelArg(kernel, 12, sizeof(cl_mem), (void *)&logSumWeightBuffer);
}


void generateSeeds()
{
    for (size_t i = 0; i < N_workers; i++)
    {
        seeds[i] = dist(mersenne_engine);
    }
    clEnqueueWriteBuffer(clInstance.commandQueue, seedBuffer, CL_TRUE, 0, N_workers*sizeof(float), (void*)seeds.data(), 0, NULL, NULL);
}

void initialize()
{
    compile();
    build();
    createKernel();
    createBuffers();
    generateSeeds();
}

void run()
{
        size_t global_work_size[1] = {N_workers};
        status = clEnqueueNDRangeKernel(clInstance.commandQueue, kernel, 1, NULL,
                                        global_work_size, NULL, 0, NULL, NULL);

        clFinish(clInstance.commandQueue);
}

std::pair<std::vector<float>, std::vector<float>> read_results()
{
    std::vector<float> param_res(paramBufferLength);
    std::vector<float> logSumWeight_res(N_MCMC+1);
    status = clEnqueueReadBuffer(clInstance.commandQueue, paramBuffer, CL_TRUE, 0,
                                    paramBufferLength*sizeof(float), param_res.data(), 0, NULL, NULL);
    assert(status == CL_SUCCESS);
    status = clEnqueueReadBuffer(clInstance.commandQueue, paramBuffer, CL_TRUE, 0,
                                    (N_MCMC+1)*sizeof(float), logSumWeight_res.data(), 0, NULL, NULL);
    assert(status == CL_SUCCESS);
    return std::make_pair(param_res, logSumWeight_res);
}

~MCMC_SIR_Kernel()
{
    status = clReleaseMemObject(seedBuffer);
    status = clReleaseMemObject(paramBuffer);
    status = clReleaseMemObject(logSumWeightBuffer);
    status = clReleaseMemObject(x0_Buffer);
    status = clReleaseMemObject(yBuffer);
    status = clReleaseMemObject(initParamBuffer);
    status = clReleaseMemObject(prop_std_Buffer);
    status = clReleaseKernel(kernel);
}

};

#endif