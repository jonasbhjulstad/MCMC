
#ifndef SMC_SAMPLER_HH

#define SMC_SAMPLER_HH 

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <cstring>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <chrono>
#include <fstream>
#include <limits>
#include <math.h>
#include "Distributions.hh"
#include "Particle.hh"
#include "SMC_Model.hh"


enum ResampleType
{
    SMC_RESAMPLE_MULTINOMIAL = 0,
    SMC_RESAMPLE_RESIDUAL,
    SMC_RESAMPLE_STRATIFIED,
    SMC_RESAMPLE_SYSTEMATIC
};
enum Iterflag{T_MAX_REACHED, ITERATE_SUCCESS};

namespace SMC
{

    /// A template class for an interacting particle system suitable for SMC sampling
    class SMC_Sampler
    {
    private:
        ///The current evolution time of the system.
        long T;
        ///The resampling mode which is to be employed.
        ResampleType rtResampleMode;
        ///The effective sample size at which resampling should be used.
        double dResampleThreshold;
        ///Structure used internally for resampling.
        double *dRSWeights;
        ///Structure used internally for resampling.
        unsigned int *uRSCount;
        ///Structure used internally for resampling.
        unsigned int *uRSIndices;
        ///The number of smc moves which have been accepted during this iteration
        int nAccepted;
        ///A flag which tracks whether the ensemble was resampled during this iteration
        int nResampled;
    protected:
        //
        long T_max;
        //Current parameter proposal
        double* propParam;
        //Number of particles in the system.
        long N_particles;
        //Particles
        Particle* pParticles;
        //Random distribution sampler
        Rng* G;
        //Model class
        Model* m_Model;
    public:
        
        //Number of model parameters (Is kept public to make MCMC copy-construction work)
        long N_param;
        ///Create an particle system containing lSize unInitialized particles with the specified mode.
        SMC_Sampler(Model*, const long&, const double*, const long &, const long &);
        //Copy Constructor:
        SMC_Sampler(SMC_Sampler*);
        ///Dispose of a sampler.
        ~SMC_Sampler();
        ///Calculates and Returns the Effective Sample Size.
        double GetESS(void);
        ///Initialize the sampler and its constituent particles.
        void InitializeParticles();
        ///Perform one time step iterate
        int Iterate(double & LogAvgWeight);
        ///Move the particle set by proposing an applying an appropriate move to each particle.
        void MoveParticles(void);
        ///Resample the particle set
        void Resample(ResampleType lMode);
        //
        void Normalize_Accumulate_Weights();
        //
        void SetResampleParams(ResampleType rtMode, double dThreshold);
        //Reset particles
        void ResetParticles();
    };
}

#endif
