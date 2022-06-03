#include "MCMC_Sampler.hh"

namespace SMC
{

//Instantiate MCMC-Sampler with COPY of SMC-Sampler instance
MCMC_Sampler::MCMC_Sampler(SMC_Sampler* pSMC, const long & NMCMC):
SMC_Sampler(pSMC),
N_MCMC(NMCMC),
N_param(pSMC->N_param),
T_MCMC(0)
{
    params = new double*[NMCMC];
    for (int i = 0; i < NMCMC; i++)
    {
        params[i] = new double[N_param];
    }
    LogWeightSums = new double[NMCMC];

}

//
MCMC_Sampler::MCMC_Sampler(Model *M, const long & Nparticles, const double* param_init, const long & Nparam, const long & Tmax, const long& NMCMC):
SMC_Sampler(M, Nparticles, param_init, Nparam, Tmax),
N_MCMC(NMCMC), 
N_param(Nparam),
T_MCMC(0)
{
    params = new double*[N_MCMC];
    for (int i = 0; i < NMCMC; i++)
    {
        params[i] = new double[N_param];
    }
    LogWeightSums = new double[N_MCMC];
}

MCMC_Sampler::~MCMC_Sampler()
{
    if (params)
    {
        for (int i=0; i < N_MCMC; i++){
            if (params[i])
                delete [] params[i];}
        delete[] params;
    }
    if(LogWeightSums)
        delete[] LogWeightSums;
}

void MCMC_Sampler::IterateMCMC()
{
    double particleLogAvgWeight = 0;
    LogWeightSums[T_MCMC] = 0;

    while (Iterate(particleLogAvgWeight) != T_MAX_REACHED)
    {
        LogWeightSums[T_MCMC]+=particleLogAvgWeight;
    }

    Metropolis();
    ResetParticles();
    T_MCMC++;
}

int MCMC_Sampler::Metropolis(void)
{
    double prevParam[N_param];
    if (T_MCMC == 0){
        memcpy(prevParam, propParam, sizeof(double)*N_param);
        memcpy(params[T_MCMC], propParam, sizeof(double)*N_param);}
    else{
        memcpy(prevParam, params[T_MCMC-1], sizeof(double)*N_param);}

    double ll_prev = (T_MCMC == 0) ? -std::numeric_limits<double>::infinity(): LogWeightSums[T_MCMC-1];
    double ll_prop = LogWeightSums[T_MCMC];
    //Did the proposal likelihood improve fit?
    double alpha_prop = exp(ll_prop - ll_prev);
    double alpha_Metropolis = (1 < alpha_prop) ? 1 : alpha_prop;

    int accept=0;
    //Accept the proposal
    if (G->Uniform(0,1) < alpha_Metropolis)
    {
        LogWeightSums[T_MCMC] = ll_prop;
        memcpy(params[T_MCMC], propParam, sizeof(double)*N_param);
        accept++;
    }
    //Reject the proposal
    else{
        LogWeightSums[T_MCMC] = ll_prev;
        memcpy(params[T_MCMC], prevParam, sizeof(double) * N_param);
    }

    //Draw new proposal sample
    m_Model->ProposalSample(params[T_MCMC], propParam);

    //Ensure that parameters are positive by running abs(propParam)
    for (int k = 0; k < N_param; k++)
    {
        propParam[k] = std::abs(propParam[k]);
    }

    return accept;
}   


void MCMC_Sampler::StreamParameters(std::fstream &f)
{
    for (int i=0; i < T_MCMC; i++)
    {
        for (int k=0; k < (N_param-1); k++)
        {
            f << params[i][k] << ",";
        }
        f << params[i][N_param-1] << std::endl;
    }
}


void MCMC_Sampler::StreamWeights(std::fstream &f)
{
    for (int i=0; i < T_MCMC; i++)
    {
        f << LogWeightSums[i] << std::endl;
    }
}
}