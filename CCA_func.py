# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from scipy import stats

def CCA_corrcoeff(X, Y, n_components):
    n_components = 3
    cca = CCA(n_components)
    U, V = cca.fit_transform(X, Y)

    X_mean = np.subtract(X, X.mean(axis=0))
    Y_mean = np.subtract(Y, Y.mean(axis=0))

    A = np.linalg.solve(X_mean.T.dot(X_mean), X_mean.T.dot(U))
    B = np.linalg.solve(Y_mean.T.dot(Y_mean), Y_mean.T.dot(V))
    return A, B, U, V


num_samples = 100
sampling_rate = 128

## Create Sine Vectors of 3 frequencies
time = np.arange(0,(num_samples)/sampling_rate, 1/sampling_rate)
base_sin10 = np.sin(2*np.pi*time*10)
base_sin15 = np.sin(2*np.pi*time*15)
base_sin20 = np.sin(2*np.pi*time*20)
base_sin12 =  np.sin(2*np.pi*time*12)

Y = np.stack(((base_sin12, base_sin15, base_sin20))).T
Y1 = np.stack(((base_sin10))).T
Y2 = np.stack(((base_sin12))).T
Y3 = np.stack(((base_sin15))).T

## Create fake datasets
sin_noise10 = base_sin10 + 2*np.random.randn(num_samples)
sin_noise12 = base_sin12 + 2*np.random.randn(num_samples)
sin_noise15 = base_sin15 + 2*np.random.randn(num_samples)

X1 = 10*sin_noise10 +    sin_noise12 + sin_noise15
X2 =    sin_noise10 + 10*sin_noise12 + sin_noise15
X3 = 20*sin_noise10 +    sin_noise12 + sin_noise15

X = np.stack(((X1, X2, X3))).T

## CCA
A, B, U, V = CCA_corrcoeff(X, Y, 3)
# A1, B1, U1, V1 = CCA_corrcoeff(X, Y1, 1)



cca = CCA(1)
U1, V1 = cca.fit_transform(X, Y1)
cca = CCA(1)
U2, V2 = cca.fit_transform(X, Y2)
cca = CCA(1)
U3, V3 = cca.fit_transform(X, Y3)


print(np.diag(np.dot(U1.T, V1)))
print(np.diag(np.dot(U2.T, V2)))
print(np.diag(np.dot(U3.T, V3)))
