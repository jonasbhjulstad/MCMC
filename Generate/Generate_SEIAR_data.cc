

#include <iostream>
#include <vector>
#include <fstream>
#include <boost/numeric/odeint.hpp>
#include <boost/range/iterator_range.hpp>



//[ rhs_function
/* The type of container used to hold the state vector */
typedef std::vector< double > state_type;

const double p_alpha = 1.0/9;
const double R0 = 1.2;
const double p_beta = p_alpha*R0;
const double p_gamma =1./3;
const double p_p = 0.5;
const double p_mu = p_alpha;
double N_pop = 1e4;
double I0 = 1000;
double DT = 1;
long N_observations = 200;

/* The rhs of x' = f(x) */
void f_SEIAR( const state_type &x , state_type &dxdt , const double /* t */ )
{
    dxdt[0] = -p_beta/N_pop*x[0]*(x[2] + x[3]);
    dxdt[1] = p_beta/N_pop*x[0]*(x[2] + x[3]) - p_gamma*x[1];
    dxdt[2] = p_p*p_gamma*x[1] - p_alpha*x[2];
    dxdt[3] = (1-p_p)*p_gamma*x[1]-p_mu*x[3];
    dxdt[4] = p_mu*x[3] + p_alpha*x[2];
}



//[ integrate_observer
struct push_back_state_and_time
{
    std::vector< state_type >& m_states;
    std::vector< double >& m_times;

    push_back_state_and_time( std::vector< state_type > &states , std::vector< double > &times )
    : m_states( states ) , m_times( times ) { }

    void operator()( const state_type &x , double t )
    {
        m_states.push_back( x );
        m_times.push_back( t );
    }
};
//]


int main(int /* argc */ , char** /* argv */ )
{
    using namespace std;
    using namespace boost::numeric::odeint;



    //[ state_initialization
    state_type x(5);
    x[0] = N_pop - I0; // start at x=1.0, p=0.0
    x[1] = 0;
    x[2] = I0;
    x[3] = 0;
    x[4] = 0;
    state_type x0(5);
    x0 = x;
    //]

    //[ integrate_observ
    vector<state_type> x_vec;
    vector<double> times;





    //[ define_adapt_stepper
    typedef runge_kutta_cash_karp54< state_type > error_stepper_type;
    //]


    times.clear();
    x_vec.clear();
    //[ integrate_adapt
    typedef controlled_runge_kutta< error_stepper_type > controlled_stepper_type;
    controlled_stepper_type controlled_stepper;
    integrate_n_steps( controlled_stepper , f_SEIAR , x0 , 0.0 , DT, N_observations, push_back_state_and_time(x_vec, times));
    //]

    std::ofstream yFile;
    yFile.open("/home/deb/Documents/MCMC_git/Custom_MCMC/Data/SEIAR_I0_1000.csv");

    yFile << N_observations << '\n' << DT << '\n';
    for (size_t i=0; i<=N_observations; i++)
    {
        yFile << x_vec[i][0] << ',' << x_vec[i][1] << ',' << x_vec[i][2] << ',' << x_vec[i][3] << "," << x_vec[i][4] << '\n';
    }

    yFile.close();


}

 
