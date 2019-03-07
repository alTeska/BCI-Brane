# Imports
import numpy as np
import rcca
import matplotlib.pyplot as plt

num_samples = 700
sampling_rate = 128

time = np.arange(0,(num_samples)/sampling_rate, 1/sampling_rate)
base_sin12 =  np.sin(2*np.pi*time*12)
base_sin15 =  np.sin(2*np.pi*time*15)
base_sin20 =  np.sin(2*np.pi*time*20)

data2 = np.stack(((base_sin12, base_sin15, base_sin20))).T

sin_noise12 = base_sin12 + 2*np.random.randn(num_samples)
sin_noise15 = base_sin15 + 2*np.random.randn(num_samples)
sin_noise20 = base_sin20 + 2*np.random.randn(num_samples)

X1 = 10*sin_noise12 +    sin_noise15 + sin_noise20
X2 =    sin_noise12 + 10*sin_noise15 + sin_noise20
X3 = 20*sin_noise12 +    sin_noise15 + sin_noise20

data1 = np.stack(((X1, X2, X3))).T


# Split each dataset into two halves: training set and test set
train1 = data1[:int(num_samples/2)]
train2 = data2[:int(num_samples/2)]
test1 = data1[int(num_samples/2):]
test2 = data2[int(num_samples/2):]

# nComponents =
# cca = rcca.CCA(kernelcca = False, reg = 0., numCC = nComponents)
cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 3)

print(cca.train([train1, train2]))

print(cca.cancorrs)
print(np.shape(cca.comps))
# print(rcca._listcorr(train1))





# print ('''The canonical correlations are:\n
# Component 1: %.02f\n
# Component 2: %.02f\n
# Component 3: %.02f\n
# ''' % tuple(cca.cancorrs))
