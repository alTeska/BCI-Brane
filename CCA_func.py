# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from scipy import stats

def CCA_corrcoeff(X, Y, n_components):
    n_components = 3
    cca = CCA(n_components)
    U, V = cca.fit_transform(X, Y)

    X_calc = np.subtract(X, X.mean(axis=0))
    Y_calc = np.subtract(Y, Y.mean(axis=0))

    A = np.linalg.solve(X_calc.T.dot(X_calc), X_calc.T.dot(U))
    B = np.linalg.solve(Y_calc.T.dot(Y_calc), Y_calc.T.dot(V))
    return A, B, U, V


num_samples = 700
sampling_rate = 128

## Create Sine Vectors of 3 frequencies
time = np.arange(0,(num_samples)/sampling_rate, 1/sampling_rate)
base_sin10 = np.sin(2*np.pi*time*10)
base_sin15 = np.sin(2*np.pi*time*15)
base_sin20 = np.sin(2*np.pi*time*20)
base_sin12 =  np.sin(2*np.pi*time*12)

Y = np.stack(((base_sin10, base_sin15, base_sin20))).T

## Create fake datasets
sin_noise10 = base_sin10 + 2*np.random.randn(num_samples)
sin_noise15 = base_sin15 + 2*np.random.randn(num_samples)
sin_noise20 = base_sin20 + 2*np.random.randn(num_samples)

X1 = 10*sin_noise10 +    sin_noise15 + sin_noise20
X2 =    sin_noise10 + 10*sin_noise15 + sin_noise20
X3 = 20*sin_noise10 +    sin_noise15 + sin_noise20

X = np.stack(((X1, X2, X3))).T

## CCA
A, B, _, _ = CCA_corrcoeff(X, Y, 3)

print(A,'\n')
print(B,'\n')
