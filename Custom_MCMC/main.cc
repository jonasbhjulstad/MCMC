#include <iostream>
#include <cmath>
#include <string.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <MCMC_Sampler.hpp>
#include "include/simfunctions.hh"


#define N_OBSERVATIONS_MAX 400

using namespace std;
long load_data(char const*, double** &data_ptr);
const char fpath[] = "/home/deb/Documents/MCMC_git/Custom_MCMC/Data/SIR_y.csv";
const char paramPath[] = "/home/deb/Documents/MCMC_git/Custom_MCMC/Data/param_out.csv";
const char weightPath[] = "/home/deb/Documents/MCMC_git/Custom_MCMC/Data/weight_out.csv";


///Annealing schedule constant
double dSchedule = 30.0;
///Rare event threshold
double dThreshold = 5.0;

long lNumber = 1000;

void b();
double N_pop = 1e8;
double I0 = 1e7;
double alpha = 1.0/9;
double R0 = 1.2;
double beta = R0*alpha;
double x0[] = {N_pop - I0, I0, 0};
long Nx = 3;
double std_ll = 1000000;
double dt = 1.0;
double** data_ptr;
double y[N_OBSERVATIONS_MAX];
long N_iterates;
long N_MCMC = 1000000;
long N_param = 4;
long N_sysparam = 2;
double param[] = {alpha, beta, N_pop, dt};

double prop_std[] = {.1*alpha, .1*beta};
double prop_mu[] = {alpha, beta};

int main(int argc, char** argv)
{
  N_iterates = load_data(fpath, data_ptr);

  for (int i = 0; i < N_iterates; i++)
  {
    y[i] = data_ptr[i][1];
  }
  SIR_Model SIR(N_sysparam, Nx, N_iterates, y, x0, prop_std);

  smc::sampler<pSIR> Sampler(lNumber, N_iterates,N_MCMC, N_param, param, N_sysparam);
  Sampler.SetSampleFunctions(SIR.init, SIR.step, SIR.proposal_sample, SIR.reset);


  Sampler.SetResampleParams(SMC_RESAMPLE_STRATIFIED, 0.99995);

  Sampler.Initialize(param);
  cout << "Running MCMC-algorithm.." << endl;
  for (int i = 0; i < N_MCMC; i++)
  {
    Sampler.IterateMCMC();
  }
  b();
  fstream f;
  f.open(paramPath, fstream::out);
  Sampler.StreamParameters(f);
  f.close();
  f.open(weightPath, fstream::out);
  Sampler.StreamWeights(f);
  f.close();
}


long load_data(char const * szName, double** &data_ptr)
{
  fstream f;
  f.open(szName,ios::in);
  string line;
  getline(f, line);
  long lIterates = stol(line, NULL, 10);
  double** data = new double*[lIterates];
  for (int i=0; i < lIterates; i++)
  {
    data[i] = new double[3];
  }
  vector<string> row;
  string word;
  // cout << "writing data.." << endl;
  for(long i = 0; i < lIterates; ++i)
    {
      
      getline(f, line);
      stringstream s(line);
      for (int k = 0; k < 3; k++)
      {
        getline(s, word, ',');
        data[i][k] = stod(word);
        // cout << data[i][k] << endl;
      }
    }
  f.close();
  // cout << "writing complete!" << endl;
  data_ptr = data;
  return lIterates;
}
void b()
{

}