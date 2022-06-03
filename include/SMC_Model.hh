#ifndef SMC_MODEL_H
#define SMC_MODEL_H
#include <gsl/gsl_rng.h>
#include "Particle.hh"
#include "Distributions.hh"
namespace SMC{
class Model
{
    public:

    //Should contain allocation necessary to run
    virtual void Init(double*&, long &) = 0;
    //Should integrate the state & update particle weight
    virtual void Step(const long &lTime, double* X, const double *param)=0;

    //Update the parameter proposal:
    virtual void ProposalSample(const double*, double*) = 0;
    //Likelihood required for particle weight updates
    virtual double LogLikelihood(const double *, const long &) = 0;
    //Retrieve Random Number Generator from model:
    virtual Rng* RngPtr() = 0;
    //Do measures necessary to start a new SMC-run:
    virtual void Reset(double*) = 0;
};
}


#endif