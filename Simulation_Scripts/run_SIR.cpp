#include <iostream>
#include <cmath>
#include <string.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <CLG.hpp>
#include <csv_reader.hpp>
#include <random>
#include <CLCPP_MCMC_FilePaths.hpp>
#include <CLCPP_PRNG.hpp>
//Setup defined here!
#include "Parameter_Configuration.hpp"

using namespace std;
#ifndef SPIRV_COMPILER
#define SPIRV_COMPILER "clang"
#endif

std::string kernel_file = std::string(CLCPP_MCMC_KERNEL_SOURCE_DIR) + "/MCMC_Epidemiological";
const long N_ODE_params = 2;
const long Nx = 3;
// extern void load_data(std::string, double**, double*, const long &);

void compile_kernel(size_t N_observations, size_t N_MCMC_ITERATIONS, size_t N_particles, CLG_PRNG_TYPE prng_type = CLG_PRNG_TYPE_KISS99)
{
    std::string spirv_compile_command = std::string(SPIRV_COMPILER) + " -c -cl-kernel-arg-info -cl-std=clc++2021 " + kernel_file + ".clcpp -o " + kernel_file + ".spv";
    std::string preprocessor_definitions = " -D PRNG_GENERATOR=" + CLG_PRNG_class_strmap.at(prng_type) + 
     + " -D N_OBSERVATIONS=" + std::to_string(N_observations)
     + " -D N_MCMC_ITERATIONS=" + std::to_string(N_MCMC_ITERATIONS)
     + " -D N_PARTICLES=" + std::to_string(N_particles);

    std::string kernel_include_directories = " -I " + std::string(CLCPP_MCMC_KERNEL_DECOMPOSITIONS_DIR) + 
    " " + std::string(CLCPP_PRNG_INCLUDE) + 
    " -I " + std::string(CLG_KERNEL_DIR) + "/Epidemiological/";
    std::cout << CLCPP_PRNG_INCLUDE << std::endl;
    std::cout << spirv_compile_command + preprocessor_definitions + kernel_include_directories << std::endl;

    int res = std::system((spirv_compile_command + preprocessor_definitions + kernel_include_directories).c_str());
    std::cout << "system Error code: " << res << std::endl;
}

int main(int argc, char** argv)
{

  N_observations = load_data(std::string(CLCPP_MCMC_DATA_DIR) + "/SIR_I0_10.csv", data_ptr, &dt, Nx);
  for (int i= 0; i < N_nu; i++)
  {
    nu_list[i] = i;
  }

  for (int i = 0; i < N_observations; i++)
  {
    y[i] = data_ptr[i][1];
  }


    CLG_Instance clInstance = clDefaultInitialize();

    std::string kernel_file = std::string() + "SIR_Compute_Stochastic";

    constexpr size_t N_observations = 100;
    compile_kernel(N_observations, N_MCMC, N_particles);

    int err = 0;
    std::string programBinary = convertToString((kernel_file + ".spv").c_str());
    long unsigned int programSize = sizeof(char)*programBinary.length();


    clInstance.program = clCreateProgramWithIL(clInstance.context, (const void*) programBinary.data(), sizeof(char)*programBinary.length(), &err);
    assert(err == CL_SUCCESS);

    std::string build_options = "";
    /*Step 6: Build program. */
    int status = clBuildProgram(clInstance.program, 1, clInstance.device_ids.data(), build_options.c_str(), NULL, NULL);
    CLG_print_build_log(status, clInstance);
    /* Parameter/Buffer initialization */
    constexpr size_t N_trajectories = 10 * 1024;
    size_t N_SingleRun_Trajectories = (N_trajectories > clInstance.max_work_group_size) ? clInstance.max_work_group_size : N_trajectories;

    size_t N_runs = std::ceil(N_trajectories / N_SingleRun_Trajectories);

    cl_ulong seeds[N_SingleRun_Trajectories];


    size_t seedBufferSize = N_SingleRun_Trajectories * sizeof(cl_ulong);

    cl_mem seedBuffer = clCreateBuffer(clInstance.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       seedBufferSize, (void *)seeds, &err);

    cl_mem x0_SIR_Buffer = clCreateBuffer(clInstance.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(float)*Nx, (void *)x0_SIR, &err);
    const size_t paramBufferLength = N_param_SIR*(N_MCMC+1);

    cl_mem paramBuffer = clCreateBuffer(clInstance.context, CL_MEM_WRITE_ONLY,
                                         paramBufferLength* sizeof(float), NULL, &err);
    const size_t logSumWeightBufferLength = (N_MCMC+1);
    cl_mem logSumWeightBuffer = clCreateBuffer(clInstance.context, CL_MEM_WRITE_ONLY,
                                                logSumWeightBufferLength*sizeof(float), NULL, &err);

    /*Step 8: Create kernel object */
    cl_kernel kernel = clCreateKernel(clInstance.program, "SIR_Compute_Stochastic", &err);
    assert(err == CL_SUCCESS);
    /*Step 9: Sets Kernel arguments.*/
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&seedBuffer);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&x0_SIR_Buffer);
    status = clSetKernelArg(kernel, 2, sizeof(float), (void *)&dt);
    status = clSetKernelArg(kernel, 3, sizeof(float), (void *)&prop_std);
    status = clSetKernelArg(kernel, 4, sizeof(float), (void *)&ll_std);
    status = clSetKernelArg(kernel, 5, sizeof(float), &nu_I);
    status = clSetKernelArg(kernel, 6, sizeof(float), &nu_R);
    status = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&paramBuffer);
    status = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&logSumWeightBuffer);


    assert(status == CL_SUCCESS);

    float param_res[paramBufferLength];
    float logSumWeight_res[logSumWeightBufferLength];

    std::ofstream outfile(std::string(CLCPP_MCMC_DATA_DIR) + "/SIR_Stochastic/x_traj.csv");
    std::mt19937 rng;
    std::uniform_int_distribution<cl_ulong> dist(0, UINT64_MAX);

    for (int i = 0; i < N_runs; i++)
    {
        for (int j = 0; j < N_SingleRun_Trajectories; j++)
        {
            seeds[j] = dist(rng);
        }
        clEnqueueWriteBuffer(clInstance.commandQueue, seedBuffer, CL_TRUE, 0, seedBufferSize, seeds, 0, NULL, NULL);
        std::cout << "Run " << i << std::endl;
        size_t global_work_size[1] = {N_SingleRun_Trajectories};
        status = clEnqueueNDRangeKernel(clInstance.commandQueue, kernel, 1, NULL,
                                        global_work_size, NULL, 0, NULL, NULL);

        clFinish(clInstance.commandQueue);

        status = clEnqueueReadBuffer(clInstance.commandQueue, paramBuffer, CL_TRUE, 0,
                                     paramBufferLength*sizeof(float), param_res, 0, NULL, NULL);
        status = clEnqueueReadBuffer(clInstance.commandQueue, paramBuffer, CL_TRUE, 0,
                                     logSumWeightBufferLength*sizeof(float), logSumWeight_res, 0, NULL, NULL);



        assert(status == CL_SUCCESS);
        // write_SIR_trajectories(outfile, x_traj.data(), Nt, N_SingleRun_Trajectories);
    }

    status = clReleaseKernel(kernel);              //Release kernel.
    status = clReleaseProgram(clInstance.program); //Release the program object.
    status = clReleaseMemObject(seedBuffer);
    status = clReleaseMemObject(paramBuffer);
    status = clReleaseMemObject(logSumWeightBuffer);
    status = clReleaseCommandQueue(clInstance.commandQueue); //Release  Command queue.
    status = clReleaseContext(clInstance.context);           //Release context.

    outfile.close();

  }



