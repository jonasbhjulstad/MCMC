#ifndef SIR_MODEL_H
#define SIR_MODEL_H
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include "MCMC_Sampler.hh"
#include "MCMC_Model.hh"
#include "rng.hh"
#include "particle.hh"
#include <vector>
#include <stdio.h>
using namespace std;



class pSIR
{
    public:
    double X[3];
};
void beta_binomial(const double & mu, const double & nu, const double & n, double & res, smc::rng* pRng);

class SIR_Model : public MCMC_Model
{
private:
    static gsl_vector *vec_prior_mu;
    static gsl_vector *vec_oldProp;
    static gsl_vector *vec_resProp;
    static gsl_matrix *mat_prior_std;
    static gsl_matrix *mat_prop_std;
    static gsl_vector *vec_prop_mu;
    static gsl_vector *vec_prior_work;

    static double ll_std;
    static double *y;
    static pSIR x0;
    static long N_iterations;
    static double dt;
    static double N_pop;
    static double nu_I;
    static double nu_R;
    static bool is_dispersed;

public:
    SIR_Model(long N_iterations, double *y_obs, double *x_init, double dt, double N_pop,  double* prop_std, double ll_std);

    static void set_dispersion_parameters(const double &nu_I_init, const double &nu_R_init);
    static void dispersion_set(long disperse_flag);

    ~SIR_Model();

    static void prior_sample(gsl_vector *res, smc::rng *pRng);

    static double prior_logLikelihood(const gsl_vector *param);


    // Corresponds to q(theta)
    static void proposal_sample(double *&res, const double *oldParam, smc::rng *pRng);


    ///A function to initialise double type markov chain-valued particles
    /// \param pRng A pointer to the random number generator which is to be used
    static smc::particle<pSIR>* init(smc::rng *pRng);


    //Calculates the next state and likelihood for that state
    static void step(long lTime, smc::particle<pSIR>* pState, double *param, smc::rng *pRng);

    //Set the state 
    static void reset(smc::particle<pSIR>* pState);

};

#endif