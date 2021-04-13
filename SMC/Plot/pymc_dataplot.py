import sys
sys.path.append(r"C:\Users\Jonas\OneDrive - NTNU\Kybernetikk og Robotikk\Epidemiske modeller\SMC")
import pickle as pck
import arviz as az
if __name__ == '__main__':

    with open('model.pck', 'rb') as f:
        model = pck.load(f)
    
    az.plot_trace(data)