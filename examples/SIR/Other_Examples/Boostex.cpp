
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



//[ rhs_function
/* The type of container used to hold the state vector */
typedef std::vector< double > state_type;

const double alpha = 1.0/9;
const double R0 = 1.2;
const double beta = alpha*R0;
double N_pop = 1e8;
double I0 = 1e7;

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

struct write_state
{
    void operator()( const state_type &x ) const
    {
        std::cout << x[0] << "\t" << x[1] << "\n";
    }
};


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



    //[ integration
    size_t steps = integrate( f_SIR ,
            x , 0.0 , 365.0 , 0.1 );
    //]



    //[ integration_class
    SIR i_SIR(alpha, beta, N_pop);
    steps = integrate( i_SIR ,
            x , 0.0 , 10.0 , 0.1 );
    //]





    //[ integrate_observ
    vector<state_type> x_vec;
    vector<double> times;

    steps = integrate( f_SIR ,
            x , 0.0 , 365.0 , 1.0 ,
            push_back_state_and_time( x_vec , times ) );

    /* output */
    for( size_t i=0; i<=steps; i++ )
    {
        cout << times[i] << '\t' << x_vec[i][0] << '\t' << x_vec[i][1] << '\n';
    }
    //]




    //[ define_const_stepper
    runge_kutta4< state_type > stepper;
    integrate_const( stepper , f_SIR , x , 0.0 , 10.0 , 0.01 );
    //]




    //[ integrate_const_loop
    const double dt = 0.01;
    for( double t=0.0 ; t<10.0 ; t+= dt )
        stepper.do_step( f_SIR , x , t , dt );
    //]




    //[ define_adapt_stepper
    typedef runge_kutta_cash_karp54< state_type > error_stepper_type;
    //]


    times.clear();
    x_vec.clear();
    x = x0;
    //[ integrate_adapt
    typedef controlled_runge_kutta< error_stepper_type > controlled_stepper_type;
    controlled_stepper_type controlled_stepper;
    steps = integrate_n_steps( controlled_stepper , f_SIR , x , 0.0 , 1.0, 365, push_back_state_and_time(x_vec, times));
    //]

    std::ofstream yFile;
    yFile.open("/home/deb/Documents/smctc-1.0/examples/SIR/Data/SIR_y.csv");

    for (size_t i=0; i<=steps; i++)
    {
        yFile << x_vec[i][0] << ',' << x_vec[i][1] << ',' << x_vec[i][2] << '\n';
    }

    yFile.close();


    {
    //[integrate_adapt_full
    double abs_err = 1.0e-4 , rel_err = 1.0e-4 , a_x = 1.0 , a_dxdt = 1.0;
    controlled_stepper_type controlled_stepper( 
        default_error_checker< double , range_algebra , default_operations >( abs_err , rel_err , a_x , a_dxdt ) );
    integrate_adaptive( controlled_stepper , f_SIR , x , 0.0 , 10.0 , 0.01 );
    //]
    }


    // //[integrate_adapt_make_controlled
    // integrate_adaptive( make_controlled< error_stepper_type >( 1.0e-10 , 1.0e-6 ) , 
    //                     f_SIR , x , 0.0 , 10.0 , 0.01 );
    // //]




    // //[integrate_adapt_make_controlled_alternative
    // integrate_adaptive( make_controlled( 1.0e-10 , 1.0e-6 , error_stepper_type() ) , 
    //                     f_SIR , x , 0.0 , 10.0 , 0.01 );
    //]

    #ifdef BOOST_NUMERIC_ODEINT_CXX11
    //[ define_const_stepper_cpp11
    {
    runge_kutta4< state_type > stepper;
    integrate_const( stepper , []( const state_type &x , state_type &dxdt , double t ) {
            dxdt[0] = -beta*x[0]*x[1]; dxdt[1] = beta*x[0]*x[1] - alpha*x[1]; dxdt[2] = alpha*x[1];}
        , x , 0.0 , 10.0 , 0.01 );
    }
    //]
    
    
    
    //[ harm_iterator_const_step]
    std::for_each( make_const_step_time_iterator_begin( stepper , f_SIR, x , 0.0 , 0.1 , 10.0 ) ,
                   make_const_step_time_iterator_end( stepper , f_SIR, x ) ,
                   []( std::pair< const state_type & , const double & > x ) {
                       cout << x.second << " " << x.first[0] << " " << x.first[1] << "\n"; } );
    //]
    #endif
    
    


}

 
