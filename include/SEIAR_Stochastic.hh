#ifndef SEIAR_MODEL_H
#define SEIAR_MODEL_H
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <math.h>
#include <cstring>
#include "Distributions.hh"
#include "Particle.hh"
#include "SMC_Model.hh"
using namespace std;

class SEIAR_Model: public SMC::Model
{
private: 
    Rng* G;
    double ll_sigma;
    double prop_sigma[5];
    double *y;
    static constexpr size_t Nx = 5;
    double x0[5];
    long N_iterations;
    double dt;
    double N_pop;
    double nu_E;
    double nu_I;
    double nu_R;
    bool is_dispersed;

public:
    SEIAR_Model(long N_iterations, double *y_obs, double *x_init, double dt, double N_pop,  double* prop_std, double ll_std);

    void set_dispersion_parameters(const double &nu_E_init, const double &nu_I_init, const double & nu_R_init);
    void dispersion_set(long disperse_flag);

    ~SEIAR_Model();

    //Should contain allocation necessary to run
    void Init(double*&, long &);
    //Should integrate the state & update particle weight
    void Step(const long &lTime, double* X, const double *param);
    //Update the parameter proposal:
    void ProposalSample(const double*, double*);
    //Likelihood required for particle weight updates
    double LogLikelihood(const double *, const long &);
    //Return Number Generator:
    Rng* RngPtr();
    //Do measures necessary to start a new SMC-run:
    void Reset(double*);    

};

#endif