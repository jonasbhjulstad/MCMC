#ifndef PARTICLE_H
#define PARTICLE_H
#include <cstring>
#include <math.h>
namespace SMC
{
class Particle
{
    private:
    //Logarithmic weight for a time instant
    double LogWeight;

    public:
    double* State;
    long N;
    double*& GetStatePtr();
    double GetLogWeight(); 
    double GetWeight();
    void CopyValue(Particle &);
    void SetLogWeight(const double& val); 
};

inline double Particle::GetLogWeight()
{
    return LogWeight;
    
}

inline double Particle::GetWeight()
{
    return exp(LogWeight);
}

inline void Particle::CopyValue(Particle & P)
{
    memcpy(State, P.State, sizeof(double)*N);
}

inline void Particle::SetLogWeight(const double& val)
{
    LogWeight = val;
}
}
#endif