#include <iostream>
#include <cmath>
#include <string.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <CLG.hpp>
#include <random>
#include <CLCPP_MCMC_FilePaths.hpp>
#include <CLCPP_PRNG.hpp>
#include <MCMC_SIR.hpp>
//Setup defined here!
#include "Parameter_Configuration.hpp"

void writeParams(std::vector<float> params, std::ofstream& f)
{
  size_t idx = 0;
    for (float param: params)
    {
      f << param;
      idx++;
      f << ((idx == 1) ? "\n" : ",");
      idx %= 2;
    }
}

void write_llSumWeights(std::vector<float> llSumWeights, std::ofstream& f)
{
    for (float llSumWeight: llSumWeights)
    {
      f << llSumWeight << "\n";
    }
}



int main(int argc, char** argv)
{
    std::string dataPath = std::string(CLCPP_MCMC_DATA_DIR) + "/SIR_I0_10.csv";

    CLG_Instance clInstance = clDefaultInitialize();
    MCMC_SIR_Kernel SIR_kernel(clInstance, dataPath, N_MCMC, N_particles, N_workers, CLG_PRNG_TYPE_KISS99);

    SIR_kernel.initialize();

    size_t N_runs = 2;

    std::ofstream paramFile, llSumWeightFile;
    std::vector<float> params, llSumWeights;
    for (int i = 0; i < N_runs; i++)
    {
      std::cout << "Run " << i << " of " << "N_runs" << std::endl;
      SIR_kernel.run();
      auto[params, llSumWeights] = SIR_kernel.read_results();
      SIR_kernel.generateSeeds();
      llSumWeightFile.open(std::string(CLCPP_MCMC_DATA_DIR) + "/SIR_Stochastic/llSumWeight_" + std::to_string(i) + ".csv");
      paramFile.open(std::string(CLCPP_MCMC_DATA_DIR) + "/SIR_Stochastic/params_" + std::to_string(i) + ".csv");
      writeParams(params, paramFile);
      write_llSumWeights(llSumWeights, llSumWeightFile);
      llSumWeightFile.close();
      paramFile.close();

      
        // write_SIR_trajectories(outfile, x_traj.data(), Nt, N_SingleRun_Trajectories);
    }

    clReleaseProgram(clInstance.program); //Release the program object.    
    clReleaseCommandQueue(clInstance.commandQueue); //Release  Command queue.
    clReleaseContext(clInstance.context);           //Release context.

}


