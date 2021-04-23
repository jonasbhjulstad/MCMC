#include <smctc.hh>
#include <SIR_funcs.hh>
// #include <cstdio> 
// #include <cstdlib>
// #include <cstring>
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <iostream>
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
  size_t lIterates;

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
    // File pointer
    fstream fin;
    // Open an existing file
    fin.open(szName, ios::in);
  
    // Read the Data from the file
    // as String Vector
    vector<string> row;
    string line, word, temp;
  
    size_t count = 0;
    while (fin >> temp) {
        count++;
        row.clear();
  
        // read an entire row and
        // store it in a string variable 'line'
        getline(fin, line);
  
        // used for breaking words
        stringstream s(line);
  
        // read every column data of a row and
        // store it in a string variable, 'word'
        while (getline(s, word)) {
  
            // add all the column data
            // of a row to a vector
            row.push_back(word);
        }
  

    }
    return count;

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
