#include "smctc.hh"
#include "SIR_funcs.hh"
#include <cstdio> 
#include <cstdlib>
#include <cstring>
#include <array>
#include <vector>
#include <unistd.h>
using namespace std;
#define PATH_MAX 1024


///The observations


double integrand_mean_x(const particle_SIR&, void*);
double integrand_mean_y(const particle_SIR&, void*);
double integrand_var_x(const particle_SIR&, void*);
double integrand_var_y(const particle_SIR&, void*);
void b();
int main(int argc, char** argv)
{

  long lNumber = 1000;
  long lIterates;
  b();
  try {
    //Load observations

    double** data = new double*[1];


    lIterates = load_data(dataPath, data);

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
void b()
{

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
