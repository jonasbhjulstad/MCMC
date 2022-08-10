

#ifndef MCMC_GENERATE_SIR_HPP
#define MCMC_GENERATE_SIR_HPP
#include <Eigen/Dense>
#include <MCMC_Distributions.hpp>
#include <cvode/cvode.h>
#include <cvode/cvode_spils.h>
#include <fstream>
#include <iostream>
#include <nvector/nvector_serial.h>
#include <oneapi/dpl/random>
#include <sundials/sundials_math.h>
#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_spgmr.h>

namespace MCMC::Integrators {

template <size_t _Nx> struct Model_Integrator {
  static constexpr size_t Nx = _Nx;
  using Vec = Eigen::Vector<double, Nx>;
  virtual Vec step(const Vec &) = 0;

  void write(const std::string &fPath, const std::vector<Vec> &data) {
    std::ofstream file(fPath);
    for (const auto &d : data) {
      file << d.transpose() << std::endl;
    }
    file.close();
  }

  std::vector<Vec> run_trajectory(const Vec &x, const size_t Nt) {
    std::vector<Vec> data(Nt);
    data[0] = x;
    for (size_t i = 1; i < Nt; i++) {
      data[i] = step(data[i - 1]);
    }
    return data;
  }
};

template <size_t _Nx, class Derived>
struct CVODE_Integrator : public Model_Integrator<_Nx> {
  using Vec = typename Model_Integrator<_Nx>::Vec;
  static constexpr size_t Nx = _Nx;
  SUNContext ctx;
  void *cvode_mem;
  realtype t_current;
  realtype dt = 0;
  realtype abs_tol;
  realtype rel_tol;
  SUNMatrix jac_mat;
  N_Vector x;
  SUNLinearSolver solver;
  CVODE_Integrator(const Vec& x0, realtype dt, realtype abs_tol = 1e-5, realtype reltol = 1e-5,
                   realtype t0 = 0)
      : dt(dt), abs_tol(abs_tol), rel_tol(reltol), t_current(t0) {

    SUNContext_Create(NULL, &ctx);
    if (check_flag((void *)ctx, "SUNContextCreate", 0))
      return;
    x = N_VNew_Serial(Nx, ctx);
    jac_mat = SUNMatNewEmpty(ctx);
    cvode_mem = CVodeCreate(CV_BDF, ctx);
    x = N_VNew_Serial(Nx, ctx);
    for (int i = 0; i < x0.size(); i++)
    {
      NV_Ith_S(x, i) = x0[i];
    }
  }

  Vec step(const Vec &x_current) {
    N_Vector x = N_VNew_Serial(Nx, ctx);
    for (int i = 0; i < Nx; i++) {
      NV_Ith_S(x, i) = x_current[i];
    }
    int flag = CVode(cvode_mem, t_current + dt, x, &t_current, CV_NORMAL);
    // copy x to Vec
    Vec x_next(Nx);
    for (int i = 0; i < Nx; i++) {
      x_next[i] = NV_Ith_S(x, i);
      if (check_flag(&flag, "CVode", 1))
        break;
    }
    return x_next;
  }

  ~CVODE_Integrator() {
    N_VDestroy(x);
    CVodeFree(&cvode_mem);
    SUNLinSolFree(solver);
    SUNContext_Free(&ctx);
  }
  int initialize_solver(CVRhsFn f, CVLsJacTimesVecFn jac) {
    int flag;
    flag = CVodeInit(cvode_mem, f, t_current, x);
    if (check_flag(&flag, "CVodeSetUserData", 1))
      return EXIT_FAILURE;
    flag = CVodeSStolerances(cvode_mem, rel_tol, abs_tol);
    if (check_flag(&flag, "CVodeSStolerances", 1))
      return EXIT_FAILURE;

    // Set linear solver as iterative
    flag = CVodeSetLinearSolver(cvode_mem, solver, jac_mat); 
    if (check_flag(&flag, "CVodeSetLinearSolver", 1))
      return EXIT_FAILURE;


    // Assign jacobian function
    flag = CVodeSetJacTimes(cvode_mem, NULL, jac);
    if (check_flag(&flag, "CVodeSetJacTimes", 1))
      return EXIT_FAILURE;
    return EXIT_SUCCESS;
  }
  private:

  static int check_flag(void *flagvalue, const char *funcname, int opt) {
    int *errflag;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated
     */
    if (opt == 0 && flagvalue == NULL) {
      fprintf(stderr,
              "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
              funcname);
      return EXIT_FAILURE;
    }

    /* Check if flag < 0 */
    else if (opt == 1) {
      errflag = (int *)flagvalue;
      if (*errflag < 0) {
        fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
                funcname, *errflag);
        return EXIT_FAILURE;
      }
    }

    /* Check if function returned NULL pointer - no memory allocated */
    else if (opt == 2 && flagvalue == NULL) {
      fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
              funcname);
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
  }
};
} // namespace MCMC::Integrators
#endif
