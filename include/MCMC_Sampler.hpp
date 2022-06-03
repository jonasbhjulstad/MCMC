#ifndef MCMC_SAMPLER_H
#define MCMC_SAMPLER_H
#include <fstream>
#include <chrono>
#include "SMC_Sampler.hpp"
namespace SMC
{
class MCMC_Sampler: public SMC_Sampler
{
private:
//Iteration Counter
long T_MCMC;
//Number of MCMC-iterations
long N_MCMC;

//List of all accepted parameter proposals:
double ** params;
//List of (unnormalized) weights from each MCMC-iteration
double* LogWeightSums;

protected:
//Number of model parameters
long N_param;

public:
//Instantiate MCMC by copying SMC-sampler
MCMC_Sampler(SMC_Sampler* pSMC, const long & N_MCMC);
//Instantiate from scratch
MCMC_Sampler(Model *M, const long & Nparticles, const double* param_init, const long & Np, const long & Tmax, const long& N_MCMC);
//
~MCMC_Sampler();
///Iterate through one chain
int Metropolis(void);
//Perform one SMC-trajectory simulation from T0->Tmax
void IterateMCMC(void);
//Dump weight list to file
void StreamWeights(std::fstream &f);
//Dump parameter list to file
void StreamParameters(std::fstream &f);
};
}
#endif