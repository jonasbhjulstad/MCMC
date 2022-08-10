import matplotlib.pyplot as plt
import numpy as np

weights = np.genfromtxt('/home/arch/Documents/SYCL_MCMC/build/test/param_list.txt', delimiter=',', dtype=float)


# the histogram of the data
n, bins, patches = plt.hist(weights, 50, density=True, facecolor='g', alpha=0.75)


plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.xlim(0., .2)
# plt.ylim(0, 1.0)
plt.grid(True)
plt.show()