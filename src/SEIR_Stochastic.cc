#include "SEIR_Stochastic.hh"

using namespace std;

SEIR_Model::SEIR_Model(long N_observations, double *y_obs, double *x_init, double DT,double Npop, double* prop_std, double llsigma)
{
  static constexpr int Nx = 4;
  G = new Rng(3);
  y = new double[N_observations];
  memcpy(y, y_obs, sizeof(double) * N_observations);
  memcpy(x0, x_init, sizeof(double)*Nx);
  ll_sigma = llsigma;
  memcpy(prop_sigma, prop_std, sizeof(double)*3);
  dt = DT;
  N_pop = Npop;
}

void SEIR_Model::set_dispersion_parameters(const double &nu_I_init, const double &nu_R_init, const double & nu_E_init){
  nu_E = nu_E_init;
  nu_I = nu_I_init;
  nu_R = nu_R_init;
  is_dispersed = 1;
}

void SEIR_Model::dispersion_set(long disperse_flag)
{
  is_dispersed = disperse_flag;
}


void SEIR_Model::ProposalSample(const double *oldParam, double* res)
{
  G->GaussianMultivariate(oldParam, prop_sigma, res);
}

void SEIR_Model::Init(double*& X, long & N)
{
    X = new double[Nx];
    memcpy(X, x0, sizeof(double)*Nx);
    N = Nx;
}  


//Calculates the next state and likelihood for that state
void SEIR_Model::Step(const long &lTime, double* X, const double *param)
{

  double alpha = param[0];
  double beta = param[1];
  double gamma = param[2];

  double p_E = 1 - exp(-beta*X[2] / N_pop * dt);
  double p_I = 1 - exp(-gamma*dt);
  double p_R = 1 - exp(-alpha * dt);

  double K_SE;
  double K_EI;
  double K_IR;

  if (is_dispersed)
  {
    K_SE = G->BetaBinomial(p_E, nu_E, X[0]);
    K_EI = G->BetaBinomial(p_I, nu_I, X[1]);
    K_IR = G->BetaBinomial(p_R, nu_R, X[2]);
  }
  else{

    K_SE = gsl_ran_poisson(G->Raw(), X[0] * p_E);
    K_EI = gsl_ran_binomial(G->Raw(), p_I, (int)X[1]);
    K_IR = gsl_ran_binomial(G->Raw(), p_R, (int)X[2]);    
  }
  double delta_x[Nx] = {-K_SE, K_SE - K_EI,K_EI - K_IR, K_IR};

  for (int i = 0; i < Nx; i++)
  {
    X[i] += delta_x[i];
  }
}

double SEIR_Model::LogLikelihood(const double* X, const long &lTime)
{
  return log(gsl_ran_gaussian_pdf(y[lTime]-X[2], ll_sigma));
}

void SEIR_Model::Reset(double * X)
{
  memcpy(X, x0, sizeof(double)*Nx);
}

Rng* SEIR_Model::RngPtr()
{
  return G;
}

SEIR_Model::~SEIR_Model()
{
  if (y)
    delete[] y;

}    