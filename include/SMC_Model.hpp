#ifndef SMC_MODEL_H
#define SMC_MODEL_H
#include <gsl/gsl_rng.h>
#include "SMC_Particle.hpp"
#include "Distributions.hpp"
namespace SMC{

template <typename Derived>
class Model
{
    public:

    //Should contain allocation necessary to run
    void Init(float*&, long &)
    {
        static_cast<Derived*>(this)->Init(state, N);
    }
    //Should integrate the state & update particle weight
    void Step(const long &lTime, float* X, const float *param)
    {
        static_cast<Derived*>(this)->Step(lTime, X, param);
    }

    //Update the parameter proposal:
    void ProposalSample(const float*, float*)
    {
        static_cast<Derived*>(this)->ProposalSample(propParam);
    }
    //Likelihood required for particle weight updates
    float LogLikelihood(const float *, const long &)
    {
        return static_cast<Derived*>(this)->LogLikelihood(state, N);
    }
    void reset_particles()  
    {
        static_cast<Derived*>(this)->reset_particles(state);
    }
};
}


#endif