#ifndef MCMC_MODEL_H
#define MCMC_MODEL_H
#include "rng.hh"
#include "particle.hh"
#include <boost/any.hpp>

class MCMC_Model
{
    public:
    //Should contain allocation necessary to run
    virtual void init(gsl_rng*) = 0;
    //Should integrate the state & update particle weight
    virtual void step(long, double*, double*, gsl_rng*) = 0;
    //Update the parameter proposal:
    virtual void proposal_sample(double* &, const double*, gsl_rng*) = 0;
    //Do measures necessary to start a new SMC-run:
    virtual void reset(double*) = 0;
};


#endif