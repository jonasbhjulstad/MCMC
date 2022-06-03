#include "SIR_Stochastic.hh"

using namespace std;

SIR_Model::SIR_Model(long N_observations, double *y_obs, double *x_init, double DT,double Npop, double* prop_std, double llsigma)
{
  Nx = 3;
  G = new Rng(2);
  y = new double[N_observations];
  memcpy(y, y_obs, sizeof(double) * N_observations);
  memcpy(x0, x_init, sizeof(double)*Nx);
  ll_sigma = llsigma;
  memcpy(prop_sigma, prop_std, sizeof(double)*2);
  dt = DT;
  N_pop = Npop;
}

void SIR_Model::set_dispersion_parameters(const double &nu_I_init, const double &nu_R_init){
  nu_I = nu_I_init;
  nu_R = nu_R_init;
  is_dispersed = 1;
}

void SIR_Model::dispersion_set(long disperse_flag)
{
  is_dispersed = disperse_flag;
}


void SIR_Model::ProposalSample(const double *oldParam, double* res)
{
  G->GaussianMultivariate(oldParam, prop_sigma, res);
}

void SIR_Model::Init(double*& X, long & N)
{
    X = new double[Nx];
    memcpy(X, x0, sizeof(double)*Nx);
    N = Nx;
}  


//Calculates the next state and likelihood for that state
void SIR_Model::Step(const long &lTime, double* X, const double *param)
{
  static constexpr size_t Nx = 3;
  double alpha = param[0];
  double beta = param[1];

  double p_I = 1 - exp(-beta*X[1] / N_pop * dt);
  double p_R = 1 - exp(-alpha * dt);

  double K_SI;
  double K_IR;

  if (is_dispersed)
  {
    K_SI = G->BetaBinomial(p_I, nu_I, X[0]);
    K_IR = G->BetaBinomial(p_R, nu_R, X[1]);
  }
  else{

    K_SI = gsl_ran_binomial(G->Raw(), p_I, (int)X[0]);
    K_IR = gsl_ran_binomial(G->Raw(), p_R, (int)X[1]);    
  }
  double delta_x[Nx] = {-K_SI, K_SI - K_IR, K_IR};

  for (int i = 0; i < Nx; i++)
  {
    X[i] += delta_x[i];
  }
}

double SIR_Model::LogLikelihood(const double* X, const long &lTime)
{
  return log(gsl_ran_gaussian_pdf(y[lTime]-X[1], ll_sigma));
}

void SIR_Model::Reset(double * X)
{
  memcpy(X, x0, sizeof(double)*Nx);
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