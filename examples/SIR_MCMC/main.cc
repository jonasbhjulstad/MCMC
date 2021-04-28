#include <iostream>
#include <cmath>
#include "simfunctions.hh"
#include <string.h>
#include <fstream>
#include <vector>
#include <sstream>

#define N_OBSERVATIONS_MAX 400

long load_data(char const*, double** &data_ptr);
const char* fpath = "/home/deb/Documents/smctc-1.0/examples/SIR/Data/SIR_y.csv";

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
double std_ll = 100;
double std_alpha = alpha/10;
double std_beta = beta/10;
double dt = 1.0;
double** data_ptr;
double y[N_OBSERVATIONS_MAX];
long N_iterates;
long N_MCMC = 1;
int main(int argc, char** argv)
{
  N_iterates = load_data(fpath, data_ptr);

  cout << N_iterates << endl;

  for (int i = 0; i < N_iterates; i++)
  {
    y[i] = data_ptr[i][2];
  }
  try{
    ///An array of move function pointers
    void (*pfMoves)(long, smc::particle<pSIR> &,smc::rng*) = fMove1;
    smc::moveset<pSIR> Moveset(fInitialise, pfMoves, fMCMC);
    smc::sampler<pSIR> Sampler(lNumber, SMC_HISTORY_RAM);

    Sampler.SetResampleParams(SMC_RESAMPLE_STRATIFIED,0.5);
    Sampler.SetMoveSet(Moveset);

    Sampler.Initialise();
    Sampler.IterateUntil(N_iterates*N_MCMC);
    
    // smc::historyelement <smc::particle <pSIR>>* helem = Sampler.GetHistory()->GetElement()->GetNext();
    // for (int i = 0; i < (N_iterates*N_MCMC-1); i++)
    // {
    //   cout << helem->GetValues()->GetValue().X[0] << endl;
    //   helem = helem->GetNext();
    // }
    ///Estimate the normalising constant of the terminal distribution
    // double zEstimate = Sampler.IntegratePathSampling(pIntegrandPS, pWidthPS, NULL) - log(2.0);
    // ///Estimate the weighting factor for the terminal distribution
    // double wEstimate = Sampler.Integrate(pIntegrandFS, NULL);
    b();
    // cout << zEstimate << " " << log(wEstimate) << " " << zEstimate + log(wEstimate) << endl;
  }
  catch(smc::exception  e)
    {
      cerr << e;
      exit(e.lCode);
    }

  return 0;
}


long load_data(char const * szName, double** &data_ptr)
{
  fstream f;
  f.open(szName,ios::in);
  string line;
  getline(f, line);
  long lIterates = stol(line, NULL, 10);
  double** data = new double*[lIterates];
  for (int i; i < lIterates; i++)
  {
    data[i] = new double[3];
  }
  vector<string> row;
  string word;
  cout << "writing data.." << endl;
  for(long i = 0; i < lIterates; ++i)
    {
      
      getline(f, line);
      // cout << "line: " << line << "liter " << lIterates << endl;
      stringstream s(line);
      for (int k = 0; k < 3; k++)
      {
        getline(s, word, ',');
        data[i][k] = stod(word);
        // cout << data[i][k] << endl;
      }
    }
  f.close();
  cout << "writing complete!" << endl;
  data_ptr = &data[0];
  return lIterates;
}
void b()
{

}