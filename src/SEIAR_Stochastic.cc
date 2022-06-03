#include "SEIAR_Stochastic.hh"

using namespace std;

SEIAR_Model::SEIAR_Model(long N_observations, double *y_obs, double *x_init, double DT,double Npop, double* prop_std, double llsigma)
{
  G = new Rng(5);
  y = new double[N_observations];
  memcpy(y, y_obs, sizeof(double) * N_observations);
  memcpy(x0, x_init, sizeof(double)*Nx);
  ll_sigma = llsigma;
  memcpy(prop_sigma, prop_std, sizeof(double)*5);
  dt = DT;
  N_pop = Npop;
}

void SEIAR_Model::set_dispersion_parameters(const double &nu_I_init, const double &nu_R_init, const double & nu_E_init){
  nu_E = nu_E_init;
  nu_I = nu_I_init;
  nu_R = nu_R_init;
  is_dispersed = 1;
}

void SEIAR_Model::dispersion_set(long disperse_flag)
{
  is_dispersed = disperse_flag;
}


void SEIAR_Model::ProposalSample(const double *oldParam, double* res)
{
  G->GaussianMultivariate(oldParam, prop_sigma, res);
}

void SEIAR_Model::Init(double*& X, long & N)
{
    X = new double[Nx];
    memcpy(X, x0, sizeof(double)*Nx);
    N = Nx;
}  


//Calculates the next state and likelihood for that state
void SEIAR_Model::Step(const long &lTime, double* X, const double *param)
{
  double alpha = param[0];
  double beta = param[1];
  double gamma = param[2];
  double p = param[3];
  double mu = param[4];

  double p_E = 1 - exp(-beta*(X[2] + X[3])/N_pop*dt);
  double p_I = 1 - exp(-p*gamma*dt);
  double p_A = 1 - exp(-(1-p)*gamma*dt);
  double p_Ri = 1 - exp(-alpha * dt);
  double p_Ra = 1 - exp(-mu * dt);

  double K_SE;
  double K_EI;
  double K_EA;
  double K_IR;
  double K_AR;

  if (is_dispersed)
  {
    K_SE = G->BetaBinomial(p_E, nu_E, X[0]);
    K_EI = G->BetaBinomial(p_I, nu_I, X[1]);
    K_EA = G->BetaBinomial(p_A, nu_I, X[1]);
    K_IR = G->BetaBinomial(p_Ri, nu_R, X[2]);
    K_AR = G->BetaBinomial(p_Ra, nu_R, X[3]);
  }
  else{

    K_SE = gsl_ran_poisson(G->Raw(), X[0] * p_E);
    K_EI = gsl_ran_binomial(G->Raw(), p_I, (int)X[1]);
    K_EA = gsl_ran_binomial(G->Raw(), p_A, (int)X[1]);
    K_IR = gsl_ran_binomial(G->Raw(), p_Ri, (int)X[2]);    
    K_AR = gsl_ran_binomial(G->Raw(), p_Ra, (int)X[3]);    
  }

  double delta_x[Nx] = {-K_SE, K_SE - K_EI - K_EA,K_EI - K_IR, K_EA - K_AR, K_IR};

  for (int i = 0; i < Nx; i++)
  {
    X[i] += delta_x[i];
  }
}

double SEIAR_Model::LogLikelihood(const double* X, const long &lTime)
{
  return log(gsl_ran_gaussian_pdf(y[lTime]-X[2], ll_sigma));
}

void SEIAR_Model::Reset(double * X)
{
  memcpy(X, x0, sizeof(double)*Nx);
}

Rng* SEIAR_Model::RngPtr()
{
  return G;
}

SEIAR_Model::~SEIAR_Model()
{
  if (y)
    delete[] y;

}    