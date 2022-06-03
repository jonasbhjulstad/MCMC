#include "SIR_Stochastic.hpp"

using namespace std;

SIR_Model::SIR_Model(constant float *y_obs, constant float *x_init, float DT, float Npop, float* prop_std, float llsigma)
{
  y = float[N_observations];
  copy_vec(y, y_obs, N_observations);
  copy_vec(x0, x_init, Nx);
  ll_sigma = llsigma;
  copy_vec(prop_sigma, prop_std, 2);
  dt = DT;
  N_pop = Npop;
}

void SIR_Model::set_dispersion_parameters(const float &nu_I_init, const float &nu_R_init){
  nu_I = nu_I_init; 
  nu_R = nu_R_init;
  is_dispersed = 1;
}

void SIR_Model::dispersion_set(long disperse_flag)
{
  is_dispersed = disperse_flag;
}


void SIR_Model::ProposalSample(const float *oldParam, float* res)
{
  G->GaussianMultivariate(oldParam, prop_sigma, res);
}



//Calculates the next state and likelihood for that state
void SIR_Model::Step(const long &lTime, float* X, const float *param)
{
  static constexpr size_t Nx = 3;
  float alpha = param[0];
  float beta = param[1];

  float p_I = 1 - exp(-beta*X[1] / N_pop * dt);
  float p_R = 1 - exp(-alpha * dt);

  float K_SI;
  float K_IR;

  if (is_dispersed)
  {
    K_SI = G->BetaBinomial(p_I, nu_I, X[0]);
    K_IR = G->BetaBinomial(p_R, nu_R, X[1]);
  }
  else{

    K_SI = gsl_ran_binomial(G->Raw(), p_I, (int)X[0]);
    K_IR = gsl_ran_binomial(G->Raw(), p_R, (int)X[1]);    
  }
  float delta_x[Nx] = {-K_SI, K_SI - K_IR, K_IR};

  for (int i = 0; i < Nx; i++)
  {
    X[i] += delta_x[i];
  }
}

float SIR_Model::LogLikelihood(const float* X, const long &lTime)
{
  return log(gsl_ran_gaussian_pdf(y[lTime]-X[1], ll_sigma));
}

void SIR_Model::Reset(float * X)
{
  memcpy(X, x0, sizeof(float)*Nx);
}

Rng* SIR_Model::RngPtr()
{
  return G;
}

SIR_Model::~SIR_Model()
{
  if (y)
    delete[] y;

}    