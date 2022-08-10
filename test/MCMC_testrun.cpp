#include <SIR_Stochastic.hpp>
#include <CL/sycl.hpp>
#include <MCMC_Sampler.hpp>
using _Engine = oneapi::dpl::ranlux48;
static _Engine engine;
static constexpr size_t Nt = 100;
std::vector<double> param_init = {1.0};
size_t N_particles = 1000;
size_t N_MCMC = 1000;
using Model = SIR_Model<double, _Engine>;
using SMC_Sampler = SMC::Sampler<Model, _Engine>;
using MCMC_Sampler = MCMC::Sampler<Model, double, _Engine>;
std::string observation_data_file = "/home/arch/Documents/SYCL_MCMC/SIR_Stochastic_Data.csv"; 
int main() {
  //read one-dimensional vector from SIR_Stochastic_Data.csv
  std::vector<double> data;
  std::ifstream file("SIR_Deterministic_Data.csv");
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
      data.push_back(std::stod(cell));
    }
  }


  // Model model();
  // double threshold = .9;
  // SMC_Sampler smc_sampler(model, threshold, N_particles,
  //                                              Nt, engine);
  // MCMC_Sampler mcmc_sampler(param_init, N_MCMC,
  //                                                        smc_sampler);
  // auto param_list = mcmc_sampler.run_chain();
  // //Write param_list to file
  // std::ofstream ofs("param_list.txt");
  // for (auto& param : param_list) {
  //   for (int i = 0; i < param.size(); i++)
  //   {
  //     ofs << param[i];
  //     if (i != 0)
  //       ofs << ",";
  //   }
  //   ofs << "\n";
  // }
  // ofs.close();

  return 0;
}
