
/*
 Copyright 2010-2012 Karsten Ahnert
 Copyright 2011-2013 Mario Mulansky
 Copyright 2013 Pascal Germroth
 Distributed under the Boost Software License, Version 1.0.
 (See accompanying file LICENSE_1_0.txt or
 copy at http://www.boost.org/LICENSE_1_0.txt)
 */


#include <iostream>
#include <vector>
#include <fstream>
#include <boost/numeric/odeint.hpp>
#include <boost/range/iterator_range.hpp>
#include <CLCPP_MCMC_FilePaths.hpp>



//[ rhs_function
/* The type of container used to hold the state vector */
typedef std::vector< double > state_type;

const double alpha = 1.0/9;
const double R0 = 1.2;
const double beta = alpha*R0;
double N_pop = 1e4;
double I0 = 1e1;
double DT = 1;
long N_observations = 1000;

/* The rhs of x' = f(x) */
void f_SIR( const state_type &x , state_type &dxdt , const double /* t */ )
{
    dxdt[0] = -beta/N_pop*x[0]*x[1];
    dxdt[1] = beta/N_pop*x[0]*x[1]-alpha*x[1];
    dxdt[2] = alpha*x[1];
}
//]



//[ rhs_class
/* The rhs of x' = f(x) defined as a class */
class SIR {

double m_alpha;
double m_beta;
double m_N_pop;

public:
    SIR( double alpha, double beta, double N_pop ) : m_alpha(alpha), m_beta(beta), m_N_pop(N_pop) { }

    void operator() ( const state_type &x , state_type &dxdt , const double /* t */ )
    {
        dxdt[0] = -beta/m_N_pop*x[0]*x[1];
        dxdt[1] = beta/m_N_pop*x[0]*x[1] - alpha*x[1];
        dxdt[2] = alpha*x[1];
    }
};
//]





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
    state_type x(3);
    x[0] = N_pop - I0; // start at x=1.0, p=0.0
    x[1] = I0;
    x[2] = 0;
    state_type x0(3);
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
    integrate_n_steps( controlled_stepper , f_SIR , x0 , 0.0 , DT, N_observations, push_back_state_and_time(x_vec, times));
    //]

    std::ofstream yFile;
    yFile.open(std::string(CLCPP_MCMC_DATA_DIR) + "/SIR_I0_10.csv");

    yFile << N_observations << '\n' << DT << '\n';
    for (size_t i=0; i<=N_observations; i++)
    {
        yFile << x_vec[i][0] << ',' << x_vec[i][1] << ',' << x_vec[i][2] << '\n';
    }

    yFile.close();


}

 
