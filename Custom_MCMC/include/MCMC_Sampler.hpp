//   SMCTC: sampler.hh
//
//   Copyright Adam Johansen, 2008.
//
//   This file is part of SMCTC.
//
//   SMCTC is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   SMCTC is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
//   along with SMCTC.  If not, see <http://www.gnu.org/licenses/>.

//! \file
//! \brief Defines the overall sampler object.
//!
//! This file defines the smc::sampler class which is used to implement entire particle systems.

#ifndef __SMC_SAMPLER_HH

#define __SMC_SAMPLER_HH 1.0

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <cstring>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#include "rng.hh"
#include "particle.hh"
///Specifiers for various resampling algorithms:
enum ResampleType
{
    SMC_RESAMPLE_MULTINOMIAL = 0,
    SMC_RESAMPLE_RESIDUAL,
    SMC_RESAMPLE_STRATIFIED,
    SMC_RESAMPLE_SYSTEMATIC
};

///Storage types for the history of the particle system.

namespace smc
{

    /// A template class for an interacting particle system suitable for SMC sampling
    template <class Space>
    class sampler
    {
    private:
        ///A random number generator.
        rng *pRng;

        ///Number of particles in the system.
        long N;
        long N_MCMC;
        ///The current evolution time of the system.
        long T;
        ///Current iteration of MCMC
        long T_MCMC;
        ///Total number of time iterations per MCMC-step
        long T_max;
        //Number of parameters to be passed to fmove()
        long N_params;
        //Model parameters
        double** params;
        //Current proposed parameters:
        double* propParam;
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
        ///Average particle weights for all time iterations in all MCMC iterations [][]
        double** pAvgLogWeights;
        ///Total weight-factor for the MCMC particle run:
        double* pMCMCLogWeights;
        ///The particles within the system.
        particle<Space> *pParticles;

        ///The number of MCMC moves which have been accepted during this iteration
        int nAccepted;
        ///A flag which tracks whether the ensemble was resampled during this iteration
        int nResampled;

        ///The function which Initializes a particle.
        particle<Space> (*pfInitialize)(rng *);
        ///The function which perform actual moves.
        void (*pfMove)(long, particle<Space> &,double*, rng *);
        ///A Markov Chain Monte Carlo move.
        int (*pfMCMC)(long, particle<Space> &, rng *);
        ///Parameter proposal function
        void (*pfProp)(double* &, const double*, rng *);

    public:
        ///Create an particle system containing lSize unInitialized particles with the specified mode.
        sampler(long lSize, long T_maximum,long N_MCMC_iterations,long N_param, double* param);
        ///Set functions
        void SetSampleFunctions(particle<Space> (*f_init)(rng *), void (*f_move)(long, particle<Space> &,double*, rng *), void (*f_prop)(double* &,const double*, rng *));
        ///Dispose of a sampler.
        ~sampler();
        ///Calculates and Returns the Effective Sample Size.
        double GetESS(void) const;
        ///Returns the number of particles within the system.
        long GetNumber(void) const { return N; }
        ///Return the value of particle n
        const Space &GetParticleValue(int n) { return pParticles[n].GetValue(); }
        ///Return the logarithmic unnormalized weight of particle n
        double GetParticleLogWeight(int n) { return pParticles[n].GetLogWeight(); }
        ///Return the unnormalized weight of particle n
        double GetParticleWeight(int n) { return pParticles[n].GetWeight(); }
        ///Returns the current evolution time of the system.
        long GetTime(void) const { return T; }
        ///Initialize the sampler and its constituent particles.
        void Initialize(double*);
        ///Perform one time iterate
        void IterateSingle(void);
        ///Iterate through one chain
        void IterateMCMC(void);
        ///Move the particle set by proposing an applying an appropriate move to each particle.
        void MoveParticles(void);
        ///Resample the particle set using the specified resmpling scheme.
        void Resample(ResampleType lMode);
        ///Set Resampling Parameters
        void SetResampleParams(ResampleType rtMode, double dThreshold);
        //Sum (total) particle weights, return sum
        double getLogWeightSum(void);
        ///Metropolis-Hastings MCMC-correction:
        int Metropolis_Hastings(void);
        ///Dump a specified particle to the specified output stream in a human readable form
        std::ostream &StreamParticle(std::ostream &os, long n);
        ///Dump the entire particle set to the specified output stream in a human readable form
        std::ostream &StreamParticles(std::ostream &os);
        ///Allow a human readable version of the sampler configuration to be produced using the stream operator.
        /// std::ostream & operator<< (std::ostream& os, sampler<Space> & s);
    };

    template <class Space>
    sampler<Space>::sampler(long lSize, long T_maximum,long N_MCMC_iterations, long N_param, double* param_init)
    {
        pRng = new rng();
        N = lSize;
        N_MCMC = N_MCMC_iterations;
        N_params = N_param;
        T_max = T_maximum;
        T_MCMC = 0;

        propParam = new double[N_param];
        memcpy(propParam, param_init, sizeof(double)*N_param);


        pAvgLogWeights = new double*[N_MCMC];
        params = new double *[N_MCMC];

        for (int i = 0; i < N_MCMC; i++)
        {
            pAvgLogWeights[i] = new double[T_max];
            params[i] = new double[N_param];
        }

        pMCMCLogWeights = new double[N_MCMC];

        pParticles = new particle<Space>[lSize];

        //Allocate some storage for internal workspaces
        dRSWeights = new double[N];
        ///Structure used internally for resampling.
        uRSCount = new unsigned[N];
        ///Structure used internally for resampling.
        uRSIndices = new unsigned[N];

        rtResampleMode = SMC_RESAMPLE_STRATIFIED;
        dResampleThreshold = 0.5 * N;
 
    }

    template <class Space>
    void sampler<Space>::SetSampleFunctions(particle<Space> (*f_init)(rng *), void (*f_move)(long, particle<Space> &, double *, rng *), void (*f_prop)(double* &,const double*, rng *))
    {
        pfInitialize = f_init;
        pfMove = f_move;
        pfProp = f_prop;
    }


    template <class Space> 
    sampler<Space>::~sampler()
    {
        delete pRng;

        if (dRSWeights)
            delete[] dRSWeights;
        if (uRSCount)
            delete[] uRSCount;
        if (uRSIndices)
            delete[] uRSIndices;
        if (propParam)
            delete[] propParam;
        if (pAvgLogWeights)
        {
            for (int i = 0; i < N_MCMC; i++)
            {
                delete[] pAvgLogWeights[i];
            }
            delete[] pAvgLogWeights;
        }
        if (params)
        {
            for (int i = 0; i < N_MCMC; i++)
            {
                delete[] params[i];
            }
            delete[] params;
        }

        if (pMCMCLogWeights)
        {
            delete[] pMCMCLogWeights;
        }
    }

    template <class Space>
    double sampler<Space>::GetESS(void) const
    {
        long double sum = 0;
        long double sumsq = 0;

        for (int i = 0; i < N; i++)
            sum += expl(pParticles[i].GetLogWeight());

        for (int i = 0; i < N; i++)
            sumsq += expl(2.0 * (pParticles[i].GetLogWeight()));

        return expl(-log(sumsq) + 2 * log(sum));
    }


    /// At present this function resets the system evolution time to 0 and calls the moveset initialisor to assign each
    /// particle in the ensemble.
    template <class Space>
    void sampler<Space>::Initialize(double* param)
    {
        T = 0;
        for (int i = 0; i < N; i++)
            pParticles[i] = pfInitialize(pRng);
        return;
    }

    template <class Space>
    void sampler<Space>::IterateSingle(void)
    {
        nAccepted = 0;

        //Move the particle set.
        MoveParticles();

        //Normalise the weights to sensible values....
        double dMaxWeight = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < N; i++)
            dMaxWeight = std::max(dMaxWeight, pParticles[i].GetLogWeight());
        for (int i = 0; i < N; i++)
            pParticles[i].SetLogWeight(pParticles[i].GetLogWeight() - (dMaxWeight));
        
        //Compute average particle weight:
        pAvgLogWeights[T_MCMC][T] = 0;
        for (int i = 0; i < N; i++)
        {
            pAvgLogWeights[T_MCMC][T] += pParticles[i].GetLogWeight();
        }
        pAvgLogWeights[T_MCMC][T]/=N;

        //Check if the ESS is below some reasonable threshold and resample if necessary.
        //A mechanism for setting this threshold is required.
        double ESS = GetESS();
        if (ESS < dResampleThreshold)
        {
            nResampled = 1;
            Resample(rtResampleMode);
        }
        else
            nResampled = 0;
        // Increment the evolution time.
        T++;
    }

    template <class Space>
    void sampler<Space>::IterateMCMC()
    {
        while (T < T_max)
            IterateSingle();

        Metropolis_Hastings();
    }

    template <class Space>
    void sampler<Space>::MoveParticles(void)
    {
        for (int i = 0; i < N; i++)
        {
            pfMove(T + 1, pParticles[i], propParam, pRng);
        }
    }

    template <class Space>
    void sampler<Space>::Resample(ResampleType lMode)
    {
        //Resampling is done in place.
        double dWeightSum = 0;
        unsigned uMultinomialCount;

        //First obtain a count of the number of children each particle has.
        switch (lMode)
        {
        case SMC_RESAMPLE_MULTINOMIAL:
            //Sample from a suitable multinomial vector
            for (int i = 0; i < N; ++i)
                dRSWeights[i] = pParticles[i].GetWeight();
            pRng->Multinomial(N, N, dRSWeights, uRSCount);
            break;

        case SMC_RESAMPLE_RESIDUAL:
            //Sample from a suitable multinomial vector and add the integer replicate
            //counts afterwards.
            dWeightSum = 0;
            for (int i = 0; i < N; ++i)
            {
                dRSWeights[i] = pParticles[i].GetWeight();
                dWeightSum += dRSWeights[i];
            }

            uMultinomialCount = N;
            for (int i = 0; i < N; ++i)
            {
                dRSWeights[i] = N * dRSWeights[i] / dWeightSum;
                uRSIndices[i] = unsigned(floor(dRSWeights[i])); //Reuse temporary storage.
                dRSWeights[i] = (dRSWeights[i] - uRSIndices[i]);
                uMultinomialCount -= uRSIndices[i];
            }
            pRng->Multinomial(uMultinomialCount, N, dRSWeights, uRSCount);
            for (int i = 0; i < N; ++i)
                uRSCount[i] += uRSIndices[i];
            break;

        case SMC_RESAMPLE_STRATIFIED:
        default:
        {
            // Procedure for stratified sampling
            dWeightSum = 0;
            double dWeightCumulative = 0;
            // Calculate the normalising constant of the weight vector
            for (int i = 0; i < N; i++)
                dWeightSum += exp(pParticles[i].GetLogWeight());
            //Generate a random number between 0 and 1/N times the sum of the weights
            double dRand = pRng->Uniform(0, 1.0 / ((double)N));

            int j = 0, k = 0;
            for (int i = 0; i < N; ++i)
                uRSCount[i] = 0;

            dWeightCumulative = exp(pParticles[0].GetLogWeight()) / dWeightSum;
            while (j < N)
            {
                while ((dWeightCumulative - dRand) > ((double)j) / ((double)N) && j < N)
                {
                    uRSCount[k]++;
                    j++;
                    dRand = pRng->Uniform(0, 1.0 / ((double)N));
                }
                k++;
                dWeightCumulative += exp(pParticles[k].GetLogWeight()) / dWeightSum;
            }
            break;
        }

        case SMC_RESAMPLE_SYSTEMATIC:
        {
            // Procedure for stratified sampling but with a common RV for each stratum
            dWeightSum = 0;
            double dWeightCumulative = 0;
            // Calculate the normalising constant of the weight vector
            for (int i = 0; i < N; i++)
                dWeightSum += exp(pParticles[i].GetLogWeight());
            //Generate a random number between 0 and 1/N times the sum of the weights
            double dRand = pRng->Uniform(0, 1.0 / ((double)N));

            int j = 0, k = 0;
            for (int i = 0; i < N; ++i)
                uRSCount[i] = 0;

            dWeightCumulative = exp(pParticles[0].GetLogWeight()) / dWeightSum;
            while (j < N)
            {
                while ((dWeightCumulative - dRand) > ((double)j) / ((double)N) && j < N)
                {
                    uRSCount[k]++;
                    j++;
                }
                k++;
                dWeightCumulative += exp(pParticles[k].GetLogWeight()) / dWeightSum;
            }
            break;
        }
        }

        //Map count to indices to allow in-place resampling
        for (unsigned int i = 0, j = 0; i < N; ++i)
        {
            if (uRSCount[i] > 0)
            {
                uRSIndices[i] = i;
                while (uRSCount[i] > 1)
                {
                    while (uRSCount[j] > 0)
                        ++j;             // find next free spot
                    uRSIndices[j++] = i; // assign index
                    --uRSCount[i];       // decrement number of remaining offsprings
                }
            }
        }

        //Perform the replication of the chosen.
        for (int i = 0; i < N; ++i)
        {
            if (uRSIndices[i] != i)
                pParticles[i].SetValue(pParticles[uRSIndices[i]].GetValue());
            pParticles[i].SetLogWeight(0);
        }
    }

    /// This function configures the resampling parameters, allowing the specification of both the resampling
    /// mode and the threshold at which resampling is used.
    ///
    /// \param rtMode The resampling mode to be used.
    /// \param dThreshold The threshold at which resampling is deemed necesary.
    ///
    /// The rtMode parameter should be set to one of the following:
    /// -# SMC_RESAMPLE_MULTINOMIAL to use multinomial resampling
    /// -# SMC_RESAMPLE_RESIDUAL to use residual resampling
    /// -# SMC_RESAMPLE_STRATIFIED to use stratified resampling
    /// -# SMC_RESAMPLE_SYSTEMATIC to use systematic resampling
    ///
    /// The dThreshold parameter can be set to a value in the range [0,1) corresponding to a fraction of the size of
    /// the particle set or it may be set to an integer corresponding to an actual effective sample size.

    template <class Space>
    void sampler<Space>::SetResampleParams(ResampleType rtMode, double dThreshold)
    {
        rtResampleMode = rtMode;
        if (dThreshold < 1)
            dResampleThreshold = dThreshold * N;
        else
            dResampleThreshold = dThreshold;
    }


    template <class Space>
    int sampler<Space>::Metropolis_Hastings(void)
    {
        double* prevParam = (T_MCMC == 0) ? propParam : params[T_MCMC];
        double ll_prev = (T_MCMC == 0) ? 0 : pMCMCLogWeights[T_MCMC-1];
        double ll_prop = 0;
        for (int i = 0; i < T_max; i++)
        {
            ll_prop += pAvgLogWeights[N_MCMC][i];
        }
        ll_prop /= T_max;

        double alpha_prop = exp(ll_prop - ll_prev);
        double alpha_Metropolis = (1 > alpha_prop) ? 1 : alpha_prop;

        T_MCMC++;

        //Accept the proposal
        if (pRng->UniformS() < alpha_Metropolis)
        {
            pMCMCLogWeights[T_MCMC] = ll_prop;
            memcpy(params[T_MCMC], propParam, sizeof(double)*N_params);
            //Draw new proposal
            pfProp(propParam,params[T_MCMC], pRng);
            //Initialize particles with a new proposal
            Initialize(propParam);
            return 0;
        }
        //Reject the proposal
        else{
            pMCMCLogWeights[T_MCMC] = ll_prop;
            memcpy(params[T_MCMC], prevParam, sizeof(double) * N_params);
            //Draw new proposal from old parameters
            pfProp(prevParam, params[T_MCMC], pRng);
            //Initialize particles with a new proposal
            Initialize(propParam);
            return 1;
        }

    }   

    template <class Space>
    std::ostream &sampler<Space>::StreamParticle(std::ostream &os, long n)
    {
        os << pParticles[n] << std::endl;
        return os;
    }

    template <class Space>
    std::ostream &sampler<Space>::StreamParticles(std::ostream &os)
    {
        for (int i = 0; i < N - 1; i++)
            os << pParticles[i] << std::endl;
        os << pParticles[N - 1] << std::endl;

        return os;
    }

}

namespace std
{
    /// Produce a human-readable display of the state of an smc::sampler class using the stream operator.

    /// \param os The output stream to which the display should be made.
    /// \param s  The sampler which is to be displayed.
    template <class Space>
    std::ostream &operator<<(std::ostream &os, smc::sampler<Space> &s)
    {
        os << "Sampler Configuration:" << std::endl;
        os << "======================" << std::endl;
        os << "Evolution Time:   " << s.GetTime() << std::endl;
        os << "Particle Set Size:" << s.GetNumber() << std::endl;
        os << std::endl;
        os << "Particle Set:" << std::endl;
        s.StreamParticles(os);
        os << std::endl;
        return os;
    }
}
#endif
