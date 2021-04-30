#ifndef SIR_MODEL_H
#define SIR_MODEL_H
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <MCMC_Sampler.hpp>
#include "rng.hh"
#include "particle.hh"
using namespace std;

class pSIR
{
    public:
    double X[3];
};

class SIR_Model
{
private:
    static gsl_vector *vec_prior_mu;
    static gsl_vector *vec_oldProp;
    static gsl_vector *vec_resProp;
    static gsl_matrix *mat_prior_std;
    static gsl_matrix *mat_prop_std;
    static gsl_vector *vec_prop_mu;
    static gsl_vector *vec_prior_work;

    static long N_param_ODE;
    static long Nx;
    static double *y;
    static double *x0;
    static long N_iterations;

public:
    SIR_Model(long N_ODE_param, long N_x, long N_iterations, double *y_obs, double *x_init, double* prop_std);

    ~SIR_Model();

    static void prior_sample(gsl_vector *res, smc::rng *pRng);

    static double prior_logLikelihood(const gsl_vector *param);


    // Corresponds to q(theta)
    static void proposal_sample(double *&res, const double *oldParam, smc::rng *pRng);


    ///A function to initialise double type markov chain-valued particles
    /// \param pRng A pointer to the random number generator which is to be used
    static smc::particle<pSIR> init(smc::rng *pRng);


    //Calculates the next state and likelihood for that state
    static void step(long lTime, smc::particle<pSIR> &pState, double *param, smc::rng *pRng);

    //Set the state 
    static void reset(smc::particle<pSIR> &pState);
  
};

#endif