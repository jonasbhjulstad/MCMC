#include "smctc.hh"
#include "SIR_funcs.hh"
#include <cstdio> 
#include <cstdlib>
#include <cstring>

using namespace std;

///The observations
double * y;
long load_data(char const * szName, double** y);

double integrand_mean_x(const particle_SIR&, void*);
double integrand_mean_y(const particle_SIR&, void*);
double integrand_var_x(const particle_SIR&, void*);
double integrand_var_y(const particle_SIR&, void*);

int main(int argc, char** argv)
{
  long lNumber = 1000;
  long lIterates;

  try {
    //Load observations
    lIterates = load_data("data.csv", &y);

    //Initialise and run the sampler
    smc::sampler<particle_SIR> Sampler(lNumber, SMC_HISTORY_NONE);  
    smc::moveset<particle_SIR> Moveset(fInitialise, fMove, NULL);

    Sampler.SetResampleParams(SMC_RESAMPLE_RESIDUAL, 0.5);
    Sampler.SetMoveSet(Moveset);
    Sampler.Initialise();
    
    for(int n=1 ; n < lIterates ; ++n) {
      Sampler.Iterate();
      
      double xm,xv,ym,yv;
      xm = Sampler.Integrate(integrand_mean_x,NULL);
      xv = Sampler.Integrate(integrand_var_x, (void*)&xm);
      
      cout << xm << "," << xv << endl;
    }
  }

  catch(smc::exception  e)
    {
      cerr << e;
      exit(e.lCode);
    }
}

long load_data(char const * szName, double** yp)
{
  FILE * fObs = fopen(szName,"rt");
  if (!fObs)
    throw SMC_EXCEPTION(SMCX_FILE_NOT_FOUND, "Error: pf assumes that the current directory contains an appropriate data file called data.csv\nThe first line should contain a constant indicating the number of data lines it contains.\nThe remaining lines should contain comma-separated pairs of x,y observations.");
  char* szBuffer = new char[1024];
  fgets(szBuffer, 1024, fObs);
  long lIterates = strtol(szBuffer, NULL, 10);

  *yp = new double[lIterates];
  
  for(long i = 0; i < lIterates; ++i)
    {
      fgets(szBuffer, 1024, fObs);
      (*yp)[i] = strtod(strtok(szBuffer, ",\r\n "), NULL);
    }
  fclose(fObs);

  delete [] szBuffer;

  return lIterates;
}

double integrand_mean_x(const particle_SIR& s, void *)
{
  return s.I;
}

double integrand_var_x(const particle_SIR& s, void* vmx)
{
  double* dmx = (double*)vmx;
  double d = (s.I - (*dmx));
  return d*d;
}
