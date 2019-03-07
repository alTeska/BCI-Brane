# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from scipy import stats

num_samples = 700
sampling_rate = 128

def corr_coeff_matrices(X, Y):
    l1 = np.shape(X)[1]
    l2 = np.shape(Y)[1]
    results = np.zeros((l1, l2))

    for n in np.arange(0,l1):
        for m in np.arange(0,l2):
            a, b = X[:,n], Y[:,m]
            results[n][m] = stats.pearsonr(a, b)[0]
    return results

## Create Sine Vectors of 3 frequencies
time = np.arange(0,(num_samples)/sampling_rate, 1/sampling_rate)
base_sin10 = np.sin(2*np.pi*time*10)
base_sin15 = np.sin(2*np.pi*time*15)
base_sin20 = np.sin(2*np.pi*time*20)
base_sin12 = np.sin(2*np.pi*time*12)


Y = np.stack(((base_sin10, base_sin15, base_sin20))).T

## Create fake datasets
sin_noise10 = base_sin10 + 2*np.random.randn(num_samples)
sin_noise15 = base_sin15 + 2*np.random.randn(num_samples)
sin_noise20 = base_sin20 + 2*np.random.randn(num_samples)

X = np.stack(((sin_noise10 + sin_noise20 + sin_noise15))).T
# X2 = np.stack(((sin_noise10, sin_noise15, sin_noise20, base_sin12))).T

## Calculate Correlation Coefficients
results   = corr_coeff_matrices(X,Y)
# resultsUV = corr_coeff_matrices(U,V)
# results2 = corr_coeff_matrices(X2, Y)


print(results)
