#include <iostream>
#include <cmath>
#include <string.h>
#include <fstream>
#include <vector>
#include <sstream>
#include "MCMC_Sampler.hh"
#include "SIR_Stochastic.hh"


#define N_OBSERVATIONS_MAX 1000
using namespace std;
long load_data(char const*, double** &data_ptr, double* DT);
const char fpath[] = "/home/deb/Documents/MCMC_git/Custom_MCMC/Data/SIR_y.csv";
const char dataDir[] = "/home/deb/Documents/MCMC_git/Custom_MCMC/Data/";


long lNumber = 100;

void b();
double N_pop = 1e4;
double I0 = 1e3;
double alpha = 1.0/9;
double R0 = 1.2;
double beta = R0*alpha;
double x0[] = {N_pop - I0, I0, 0};
long Nx = 3;
double** data_ptr;
double y[N_OBSERVATIONS_MAX];
long N_iterates;
long N_MCMC = 5000;
long N_param = 4;
long N_sysparam = 2;
double param[4] = {alpha, beta, N_pop, 0};
double prop_std[] = {.05*alpha, .05*beta};
double prop_mu[] = {alpha, beta};
double ll_std = 200;
double dt;

double nu_R = 1e-5;
const double nu_list[] = {1,2,3,4,5,6};


int main(int argc, char** argv)
{
  long N_nu = 6;


  N_iterates = load_data(fpath, data_ptr, &dt);

  for (int i = 0; i < N_iterates; i++)

  {

    y[i] = data_ptr[i][1];
  }
  double nu_I;
  for (int i=0; i < N_nu; i++)
  {
  nu_I = nu_list[i];
  SIR_Model SIR(N_iterates, 
  y, x0, dt, N_pop, prop_std, ll_std);

  SIR.set_dispersion_parameters(nu_I, nu_R);


  smc::sampler<pSIR> Sampler(lNumber, N_iterates,N_MCMC, N_param, param, N_sysparam);
  Sampler.SetSampleFunctions(SIR.init, SIR.step, SIR.proposal_sample, SIR.reset);
  Sampler.SetResampleParams(SMC_RESAMPLE_MULTINOMIAL, .95);

  Sampler.Initialize(param);
  cout << "Running MCMC-algorithm" <<  ", nu = " << nu_I << endl;
  double perc;
  for (int i = 0; i < N_MCMC; i++)
  {
    if ((i % (N_MCMC / 10)) == 0)
    cout << i << "%" << endl;
    Sampler.IterateMCMC();
  }

  string paramPath = dataDir;
  paramPath.append("param_");
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
  weightPath.append("weight_");
  weightPath.append(to_string((int)nu_I));
  weightPath.append("_");
  weightPath.append(to_string((int) ll_std));
  weightPath.append(",_");
  weightPath.append(to_string((int) N_MCMC));

  weightPath.append(".csv");

  f.open(weightPath, fstream::out);
  Sampler.StreamWeights(f);
  f.close();
  }

}


long load_data(char const * szName, double** &data_ptr, double* DT)
{
  fstream f;
  f.open(szName,ios::in);
  string line;
  getline(f, line);
  long lIterates = stol(line, NULL, 10);
  getline(f, line);
  stringstream s0(line);
  string word;
  string::size_type sz;
  DT[0] = stod(line, &sz);
  double** data = new double*[lIterates];
  for (int i=0; i < lIterates; i++)
  {
    data[i] = new double[3];
  }
  for(long i = 0; i < lIterates; ++i)
    {
      
      getline(f, line);
      stringstream s(line);
      for (int k = 0; k < 3; k++)
      {
        getline(s, word, ',');
        data[i][k] = stod(word);
      }
    }
  f.close();
  data_ptr = data;
  return lIterates;
}

