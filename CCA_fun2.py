# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from scipy import stats
from scipy.linalg import inv, eig

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
sin_noise10 = base_sin10 + .2*np.random.randn(num_samples)
sin_noise12 = base_sin12 + .2*np.random.randn(num_samples)
sin_noise15 = base_sin15 + .2*np.random.randn(num_samples)

X1 = 10*sin_noise10 +    sin_noise12 + sin_noise15
X2 =    sin_noise10 + 10*sin_noise12 + base_sin20
X3 = 20*sin_noise10 +    sin_noise12 + sin_noise15

X = np.stack(((X1, X2, X3))).T


## CCA calculations

def CCA_corrcoeff(X, Y):
    """Function calculates correlations coeffciencts r"""
    Z = np.column_stack((X, Y,))
    C = np.cov(Z.T)

    sy = np.shape(Y)[1] if Y.ndim > 1 else 1
    sx = np.shape(X)[1] if X.ndim > 1 else 1

    Cxx = C[0:sx, 0:sx] + 10**(-8) * np.eye(sx)
    Cxy = C[0:sx, sx:sx+sy]
    Cyx = Cxy.T
    Cyy = C[sx:sx+sy, sx:sx+sy] + 10**(-8) * np.eye(sy)
    invCyy = inv(Cyy)
    invCxx = inv(Cxx)

    r, Wx = eig( np.dot( np.dot(invCxx, Cxy), np.dot(invCyy, Cyx) ) )
    r = np.real(r)
    r_sqrt = np.sqrt(np.round(r, 7))

    return r, r_sqrt


r1, r1_sqrt = CCA_corrcoeff(X, Y1)
r2, r2_sqrt = CCA_corrcoeff(X, Y2)
r3, r3_sqrt = CCA_corrcoeff(X, Y3)

r = np.stack(((r1, r2, r3)))
r_sqrt = np.stack(((r1_sqrt, r2_sqrt, r3_sqrt)))
print(r)
print(r_sqrt)
