#include <iostream>
#include <cmath>
#include <string.h>
#include <fstream>
#include <vector>
#include <sstream>
#include "SIR_Stochastic.hh"
#include "SMC_Sampler.hh"
#include "MCMC_Sampler.hh"
#include "csv_reader.hh"
//Setup defined here!
#include "Parameter_Configuration.hh"

using namespace std;

const long N_ODE_params = 2;
const long Nx = 3;
// extern void load_data(std::string, double**, double*, const long &);

int main(int argc, char** argv)
{

  // Load observations into y:
  string dataPath = dataDir;
  dataPath.append("SIR_I0_");
  dataPath.append(to_string((int)I0));
  dataPath.append(".csv");
  N_observations = load_data(dataPath, data_ptr, &dt, Nx);
  for (int i= 0; i < N_nu; i++)
  {
    nu_list[i] = i;
  }

  for (int i = 0; i < N_observations; i++)
  {
    y[i] = data_ptr[i][1];
  }

  //Perform MCMC over all nu in nu_list
  for (int i=0; i < N_nu; i++)
  {
  nu_I = nu_list[i];
  SIR_Model SIR(N_observations, 
  y, x0_SIR, dt, N_pop, prop_std, ll_std);

  SIR.set_dispersion_parameters(nu_I, nu_R);
  if (nu_I == 0){
    SIR.dispersion_set(0);}
  else{
    SIR.dispersion_set(1);
  }
  SMC::SMC_Sampler smcSampler(&SIR, N_particles, propParam, N_ODE_params, N_observations);
  SMC::MCMC_Sampler Sampler(&smcSampler, N_MCMC);
  Sampler.InitializeParticles();


  cout << "Running MCMC-algorithm on SIR-Model" <<  ", nu_I = " << nu_I << endl;
  double perc = 0;
  for (int i = 0; i < N_MCMC; i++)
  {
    if ((i % (N_MCMC / 10)) == 0) {
      cout << perc << "%" << endl;
      perc+=10;}
    Sampler.IterateMCMC();
  }



  //Store weights and parameters
  string paramPath = dataDir;
  paramPath.append("SIR/param_");
  paramPath.append(to_string((int)nu_I));
  paramPath.append("_");
  paramPath.append(to_string((int) ll_std));
  paramPath.append("_");
  paramPath.append(to_string((int) N_MCMC));
  paramPath.append(".csv");
  fstream f;
  f.open(paramPath, fstream::out);
  Sampler.StreamParameters(f);
  f.close();

  string weightPath = dataDir;
  weightPath.append("SIR/weight_");
  weightPath.append(to_string((int)nu_I));
  weightPath.append("_");
  weightPath.append(to_string((int) ll_std));
  weightPath.append("_");
  weightPath.append(to_string((int) N_MCMC));

  weightPath.append(".csv");

  f.open(weightPath, fstream::out);
  Sampler.StreamWeights(f);
  f.close();
  }

}


