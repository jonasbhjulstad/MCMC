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
const char fpath[] = "/home/deb/Documents/MCMC/Custom_MCMC/Data/SIR_y.csv";

///Annealing schedule constant
double dSchedule = 30.0;
///Rare event threshold
double dThreshold = 5.0;

long lNumber = 1;

void b();
double N_pop = 1e8;
double I0 = 1e7;
double alpha = 1.0/9;
double R0 = 1.2;
double beta = R0*alpha;
double x0[] = {N_pop - I0, I0, 0};
long Nx = 3;
double std_ll = 100;
double std_alpha = alpha/10;
double std_beta = beta/10;
double dt = 1.0;
double** data_ptr;
double y[N_OBSERVATIONS_MAX];
long N_iterates;
long N_MCMC = 1;
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

  smc::sampler<pSIR> Sampler(lNumber, N_iterates,N_MCMC, N_param, param);
  Sampler.SetSampleFunctions(fInitialise, f_SIR, proposal_sample);


  Sampler.SetResampleParams(SMC_RESAMPLE_STRATIFIED, 0.5);

  Sampler.Initialize(param);
  Sampler.IterateMCMC();
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
      cout << "line: " << line << "liter " << lIterates << endl;
      stringstream s(line);
      for (int k = 0; k < 3; k++)
      {
        getline(s, word, ',');
        data[i][k] = stod(word);
        cout << data[i][k] << endl;
      }
    }
  f.close();
  cout << "writing complete!" << endl;
  data_ptr = data;
  return lIterates;
}
void b()
{

}