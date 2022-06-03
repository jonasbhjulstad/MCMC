#include "SMC_Sampler.hh"

namespace SMC
{



SMC_Sampler::SMC_Sampler(Model *M, const long & Nparticles, const double* param_init, const long & Nparams, const long & Tmax):
m_Model(M),
N_particles(Nparticles),
nResampled(0),
N_param(Nparams),
T_max(Tmax)
{
    propParam = new double[N_param];
    memcpy(propParam, param_init, sizeof(double)*N_param);

    pParticles = new Particle[N_particles];

    dRSWeights = new double[Nparticles];

    uRSCount = new unsigned[Nparticles];

    uRSIndices = new unsigned[Nparticles];

    G = m_Model->RngPtr();

}

//Copy Constructor:
SMC_Sampler::SMC_Sampler(SMC_Sampler* S):
m_Model(S->m_Model),
N_particles(S->N_particles),
nResampled(S->nResampled),
N_param(S->N_param),
T_max(S->T_max)
{
    propParam = new double[N_param];
    memcpy(propParam, S->propParam, sizeof(double)*N_param);

    pParticles = new Particle[N_particles];

    dRSWeights = new double[N_particles];

    uRSCount = new unsigned[N_particles];

    uRSIndices = new unsigned[N_particles];
    
    G = m_Model->RngPtr();
}




SMC_Sampler::~SMC_Sampler()
{
    if(dRSWeights)
        delete[] dRSWeights;
    if(uRSCount)
        delete[] uRSCount;
    if(uRSIndices)
        delete[] uRSIndices;
    if(propParam)
        delete[] propParam;
}



void SMC_Sampler::InitializeParticles()
{
    T = 0;
    for (int i = 0; i < N_particles; i++){
        m_Model->Init(pParticles[i].State, pParticles[i].N);
        pParticles[i].SetLogWeight(-std::numeric_limits<double>::infinity());}
    return;
}


void SMC_Sampler::SetResampleParams(ResampleType rtMode, double dThreshold)
{
    rtResampleMode = rtMode;
    if (dThreshold < 1)
        dResampleThreshold = dThreshold * N_particles;
    else
        dResampleThreshold = dThreshold;
    nResampled = 0;
}


void SMC_Sampler::Normalize_Accumulate_Weights()
{
    double dMaxWeight = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < N_particles; i++){
        dMaxWeight = std::max(dMaxWeight, pParticles[i].GetLogWeight());}
    for (int i = 0; i < N_particles; i++){
        pParticles[i].SetLogWeight(pParticles[i].GetLogWeight() - (dMaxWeight));
    }
}

int SMC_Sampler::Iterate(double & LogAvgWeight)
{   
    if (T < T_max)
    {
        MoveParticles();

        Normalize_Accumulate_Weights();
        //Check if the ESS is below some reasonable threshold and resample if necessary.
        //A mechanism for setting this threshold is required.
        double ESS = GetESS();
        if (ESS < dResampleThreshold)
        {
            nResampled++;
            Resample(rtResampleMode);
        }
        
        //Calculate the average logweight:
        double ll_sum = 0;
        for (int i = 0; i < N_particles; i++)
        {
            ll_sum+=pParticles[i].GetLogWeight();
        }
        LogAvgWeight = ll_sum/N_particles;

        T++;
        return ITERATE_SUCCESS;}
    else{
        return T_MAX_REACHED;}
    
}


void SMC_Sampler::MoveParticles(void)
{
    double ll;
    for (int i = 0; i < N_particles; i++)
    {
        m_Model->Step(T, pParticles[i].State, propParam);
        ll = m_Model->LogLikelihood(pParticles[i].State,T);
        //Update particle weight with loglikelihood:
        pParticles[i].SetLogWeight(ll);
    }

}

//Get Effective Sample Size
double SMC_Sampler::GetESS(void)
{
    long double sum = 0;
    long double sumsq = 0;

    for (int i = 0; i < N_particles; i++)
        sum += expl(pParticles[i].GetLogWeight());

    for (int i = 0; i < N_particles; i++)
        sumsq += expl(2.0 * (pParticles[i].GetLogWeight()));

    return expl(-log(sumsq) + 2 * log(sum));
}


void SMC_Sampler::Resample(ResampleType lMode)
{
    //Resampling is done in place.
    double dWeightSum = 0;
    unsigned uMultinomialCount;

    //First obtain a count of the number of children each particle has.
    switch (lMode)
    {

    case SMC_RESAMPLE_MULTINOMIAL:
        //Sample from a suitable multinomial vector
        for (int i = 0; i < N_particles; ++i)
            dRSWeights[i] = pParticles[i].GetWeight();
        G->Multinomial(N_particles, N_particles, dRSWeights, uRSCount);
        
        
        break;


    case SMC_RESAMPLE_RESIDUAL:
        //Sample from a suitable multinomial vector and add the integer replicate
        //counts afterwards.
        dWeightSum = 0;
        for (int i = 0; i < N_particles; ++i)
        {
            dRSWeights[i] = pParticles[i].GetWeight();
            dWeightSum += dRSWeights[i];
        }

        uMultinomialCount = N_particles;
        for (int i = 0; i < N_particles; ++i)
        {
            dRSWeights[i] = N_particles * dRSWeights[i] / dWeightSum;
            uRSIndices[i] = unsigned(floor(dRSWeights[i])); //Reuse temporary storage.
            dRSWeights[i] = (dRSWeights[i] - uRSIndices[i]);
            uMultinomialCount -= uRSIndices[i];
        }
        G->Multinomial(uMultinomialCount, N_particles, dRSWeights, uRSCount);
        for (int i = 0; i < N_particles; ++i)
            uRSCount[i] += uRSIndices[i];
        break;

    case SMC_RESAMPLE_STRATIFIED:
    default:
    {
        // Procedure for stratified sampling
        dWeightSum = 0;
        double dWeightCumulative = 0;
        // Calculate the normalising constant of the weight vector
        for (int i = 0; i < N_particles; i++)
            dWeightSum += exp(pParticles[i].GetLogWeight());
        //Generate a random number between 0 and 1/N times the sum of the weights
        double dRand = G->Uniform(0, 1.0 / ((double)N_particles));

        int j = 0, k = 0;
        for (int i = 0; i < N_particles; ++i)
            uRSCount[i] = 0;

        dWeightCumulative = exp(pParticles[0].GetLogWeight()) / dWeightSum;
        while (j < N_particles)
        {
            while ((dWeightCumulative - dRand) > ((double)j) / ((double)N_particles) && j < N_particles)
            {
                uRSCount[k]++;
                j++;
                dRand = G->Uniform(0, 1.0 / ((double)N_particles));
            }
            k++;
            if (k < N_particles)
            {

                dWeightCumulative += exp(pParticles[k].GetLogWeight()) / dWeightSum;
            }
        }
        break;
    }

    case SMC_RESAMPLE_SYSTEMATIC:
    {
        // Procedure for stratified sampling but with a common RV for each stratum
        dWeightSum = 0;
        double dWeightCumulative = 0;
        // Calculate the normalising constant of the weight vector
        for (int i = 0; i < N_particles; i++)
            dWeightSum += exp(pParticles[i].GetLogWeight());
        //Generate a random number between 0 and 1/N times the sum of the weights
        double dRand = G->Uniform(0, 1.0 / ((double)N_particles));

        int j = 0, k = 0;
        for (int i = 0; i < N_particles; ++i)
            uRSCount[i] = 0;

        dWeightCumulative = exp(pParticles[0].GetLogWeight()) / dWeightSum;
        while (j < N_particles)
        {
            while ((dWeightCumulative - dRand) > ((double)j) / ((double)N_particles) && j < N_particles)
            {
                uRSCount[k]++;
                j++;
            }
            k++;
            if (k < N_particles)
            {
                dWeightCumulative += exp(pParticles[k].GetLogWeight()) / dWeightSum;
            }
        }
        break;
    }
    }

    //Map count to indices to allow in-place resampling
    for (unsigned int i = 0, j = 0; i < N_particles; ++i)
    {
        if (uRSCount[i] > 0)
        {
            uRSIndices[i] = i;
            while (uRSCount[i] > 1)
            {
                while (uRSCount[j] > 0)
                    ++j;             // find next free spot
                uRSIndices[j++] = i; // assign index
                --uRSCount[i];       // decrement number of remaining offsprings
            }
        }
    }

    //Perform the replication of the chosen.
    for (int i = 0; i < N_particles; ++i)
    {
        if (uRSIndices[i] != i)
            pParticles[i].CopyValue(pParticles[uRSIndices[i]]);
        pParticles[i].SetLogWeight(0);
    }
}

void SMC_Sampler::ResetParticles(){

    for (int i = 0; i < N_particles; i++)
    {
        m_Model->Reset(pParticles[i].State);
        pParticles[i].SetLogWeight(0);
    }
    T = 0;
    nResampled = 0;
}

}

