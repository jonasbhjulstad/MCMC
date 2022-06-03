#include <iostream>
#include <cmath>
#include <string.h>
#include <fstream>
#include <vector>
#include <sstream>
#include "SEIR_Stochastic.hpp"
#include "SMC_Sampler.hpp"
#include "MCMC_Sampler.hpp"
#include "csv_reader.hpp"
//Setup defined here!
#include "Parameter_Configuration.hpp"

using namespace std;

const long N_ODE_params = 3;
const long Nx = 4;

int main(int argc, char** argv)
{
  // Load observations into y:
  string dataPath = dataDir;
  dataPath.append("SEIR_I0_");
  dataPath.append(to_string((int)I0));
  dataPath.append(".csv");
  N_observations = load_data(dataPath, data_ptr, &dt, Nx);
  for (int i= 0; i < N_nu; i++)
  {
    nu_list[i] = i;
  }

  for (int i = 0; i < N_observations; i++)
  {
    y[i] = data_ptr[i][2];
  }

  //Perform MCMC over all nu in nu_list
  double nu_E;
  for (int i=0; i < N_nu; i++)
  {
  nu_E = nu_list[i];
  SEIR_Model SEIR(N_observations, 
  y, x0_SEIR, dt, N_pop, prop_std, ll_std);

  SEIR.set_dispersion_parameters(nu_E, nu_I, nu_R);
  if (nu_E == 0){
    SEIR.dispersion_set(0);}
  else{
    SEIR.dispersion_set(1);
  }
  SMC::SMC_Sampler smcSampler(&SEIR, N_particles, propParam, N_ODE_params, N_observations);
  SMC::MCMC_Sampler Sampler(&smcSampler, N_MCMC);
  Sampler.InitializeParticles();


  cout << "Running MCMC-algorithm on SEIR-Model" <<  ", nu_E = " << nu_E << endl;
  double perc = 0;
  for (int i = 0; i < N_MCMC; i++)
  {
    if ((i % (N_MCMC / 10)) == 0){
      cout << perc << "%" << endl;
      perc+=10;}
    Sampler.IterateMCMC();
  }



  //Store weights and parameters
  string paramPath = dataDir;
  paramPath.append("SEIR/param_");
  paramPath.append(to_string((int)nu_E));
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
  weightPath.append("SEIR/weight_");
  weightPath.append(to_string((int)nu_E));
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


