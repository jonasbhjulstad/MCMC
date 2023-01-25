#include <pybind11/pybind11.h>
#include <Sycl_MCMC/random.hpp>

namespace py = pybind11;

PYBIND11_MODULE(, m) {
    //create submodule
    py::module m_sub = m.def_submodule("random", "");


}