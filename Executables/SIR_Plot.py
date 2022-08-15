#Plot 2 histograms of values contained in /home/arch/Documents/SYCL_MCMC/build/Executables/result.csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('/home/arch/Documents/SYCL_MCMC/build/Executables/result.csv', header=None)
data = data.values

fig, ax  = plt.subplots(2)

[ax[i].hist(data[:,i], bins=50) for i in range(2)]
plt.show()