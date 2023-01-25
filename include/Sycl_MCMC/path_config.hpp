#ifndef SYCL_MCMC_Path_CONFIG_HPP
#define SYCL_MCMC_Path_CONFIG_HPP

#include <string>
#include <sstream>

namespace SYCL_MCMC {
// const char *SYCL_MCMC_ROOT_DIR = "";
// const char *SYCL_MCMC_INCLUDE_DIR = "";
    const char *SYCL_MCMC_DATA_DIR = "/home/man/Documents/Sycl_MCMC/data";
    const char *SYCL_MCMC_LOG_DIR = "/home/man/Documents/Sycl_MCMC/log";


    std::string MC_filename(uint32_t N_pop, float p_ER, uint32_t iter, std::string network_type) {
        std::stringstream ss;
        ss << SYCL_MCMC_DATA_DIR << "/Bernoulli_" << network_type << "_MC_" << N_pop << "_" << p_ER << "/" << iter
           << ".csv";
        return ss.str();
    }
    std::string quantile_filename(uint32_t N_pop, float p_ER, uint32_t iter, std::string network_type) {
        std::stringstream ss;
        ss << SYCL_MCMC_DATA_DIR << "/Quantile_Bernoulli_" << network_type << "_MC_" << N_pop << "_" << p_ER << "/" << iter
           << ".csv";
        return ss.str();
    }
    std::string path_dirname(const std::string &fname)
    {
        size_t pos = fname.find_last_of("\\/");
        return (std::string::npos == pos)
            ? ""
            : fname.substr(0, pos);
    }


} //FROLS
#endif
