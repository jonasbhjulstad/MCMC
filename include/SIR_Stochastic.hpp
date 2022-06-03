#ifndef SIR_MODEL_H
#define SIR_MODEL_H
#include "Distributions.hpp"
#include "SMC_Particle.hpp"
#include "SMC_Model.hpp"
using namespace std;

template <size_t N_observations>
class SIR_Model: public SMC::Model<SIR_Model>
{
private: 
    float ll_sigma;
    float prop_sigma[2];
    float *y;
    long Nx;
    float x0[3];
    long N_iterations;
    float dt;
    float N_pop;
    float nu_I;
    float nu_R;
    bool is_dispersed;

public:
    SIR_Model(long N_iterations, float *y_obs, float *x_init, float dt, float N_pop,  float* prop_std, float ll_std);

    void set_dispersion_parameters(const float &nu_I_init, const float &nu_R_init);
    void dispersion_set(long disperse_flag);

    ~SIR_Model();

    //Should contain allocation necessary to run
    void Init(float*&, long &);
    //Should integrate the state & update particle weight
    void Step(const long &lTime, float* X, const float *param);
    //Update the parameter proposal:
    void ProposalSample(const float*, float*);
    //Likelihood required for particle weight updates
    float LogLikelihood(const float *, const long &);
    //Do measures necessary to start a new SMC-run:
    void Reset(float*);    

};

#endif