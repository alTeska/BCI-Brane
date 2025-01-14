# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from scipy import stats

from numpy.linalg import inv

X_matlab = np.matrix([
[-0.543235806708799,	-0.324164034219554,	1.46684257497697],
[0.415211912359980,	-0.381948666866771,	3.02454550367185],
[-0.839903972986948,	1.75084125963310,	0.538463738885295],
[0.965019281464455,	1.72381895676883,	2.84419839821259],
[-0.0421292079736393,	1.11580861279137,	1.78101155052148],
[0.304382573224879,	0.996605156059074,	1.45349658671900],
[-0.432831955644998,	2.25072295631499,	0.274940324868379],
[-0.00725203435675526,	2.60734779216350,	0.123274463321904],
[1.23192324085617,	1.74404782355863,	0.346638027671402],
[0.0388597174194594,	1.25729647234885,	3.74247272167589],
[1.42942193931314,	2.10139502465463,	3.16931487863585]
])

Y_matlab = np.matrix([
[0,	0,	0],
[0.0627905195293134,	0.0941083133185143,	0.125333233564304],
[0.125333233564304,	0.187381314585725,	0.248689887164855],
[0.187381314585725,	0.278991106039229,	0.368124552684678],
[0.248689887164855,	0.368124552684678,	0.481753674101715],
[0.309016994374947,	0.453990499739547,	0.587785252292473],
[0.368124552684678,	0.535826794978997,	0.684547105928689],
[0.425779291565073,	0.612907053652976,	0.770513242775789],
[0.481753674101715,	0.684547105928689,	0.844327925502015],
[0.535826794978997,	0.750111069630460,	0.904827052466020],
[0.587785252292473,	0.809016994374948,	0.951056516295154]])

U_matlab = np.matrix([
[1.95078878449929,	0.297894439909736,	0.0651522375439652],
[1.09812239176804,	1.36278290532600,	0.686515980572688],
[0.420222942023130,	-0.914355165504431	,-1.25965112116455],
[-1.03235285985711,	0.659418367699146,	0.226940098992198],
[0.289242427080893,	0.105968005778218	,-0.311865070453439],
[0.402406098774995,	-0.0597102862644679,	0.507401025720734],
[-0.100767402304030,	-1.21451370295726	,-0.803598240429901],
[-0.532979566913855,	-1.40000418383571	,-0.287488681277634],
[-0.225476754292238,	-0.960038058599326,	2.11257180836739],
[-0.595664441690704,	1.34328530562771	,-1.44320839236561],
[-1.67354161908841,	0.779272372820379,	0.507230354494168]])

A_matlab =  np.matrix([
[-0.348856948449181,	0.0371931844954753	,1.57055766209090],
[-0.952170326287599,	-0.288055402677248	,-0.679069640995390],
[-0.368058663927214,	0.650056919257593	,-0.592650523157563]])


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


base_sin12 =  np.sin(2*np.pi*time*12)

Y = np.stack(((base_sin10, base_sin15, base_sin20))).T

## Create fake datasets
sin_noise10 = base_sin10 + 2*np.random.randn(num_samples)
sin_noise15 = base_sin15 + 2*np.random.randn(num_samples)
sin_noise20 = base_sin20 + 2*np.random.randn(num_samples)

X = np.stack(((sin_noise10, sin_noise20, sin_noise15))).T
X2 = np.stack(((sin_noise10, sin_noise15, sin_noise20, base_sin12))).T

## CCA
n_components = 3
cca = CCA(n_components)
U, V = cca.fit_transform(X_matlab, Y_matlab)

X_mean = X_matlab.mean(axis=0)
U_check = np.subtract(X_matlab, X_mean) * A_matlab
A = np.dot(U_matlab.T, np.subtract(X_matlab, X_mean))

X_calc = np.subtract(X_matlab, X_mean)

B = X_calc
b = U_matlab

A = np.linalg.solve(B.T.dot(B), B.T.dot(b))
print(c)
