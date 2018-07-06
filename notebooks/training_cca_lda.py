import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, integrate
import scipy.fftpack
from scipy.signal import butter, lfilter, filtfilt


from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC

from filtering_cca import * #cca_corrs
import pickle

# Load data
path = '../data/raw/'
sampling_rate = 128

#Load and shape the data
fnames = glob(path+'*.csv') # get paths and save them
fnames.sort()
col_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'label']
data = {}
for i, name in enumerate(fnames):
    data[i] = pd.read_csv(name, names=col_names)
    data[i]['id'] = i
data_all = pd.DataFrame()
for i in np.arange(0,3):
    data_all = pd.concat([data_all, data[i]], axis=0, ignore_index=True)
yagmur = data[0]


# Get the turn on Points for Epoching
threshold = 0.5
idxOFF = np.argwhere(yagmur.label < threshold)
idxON = np.argwhere(yagmur.label > threshold)
x_alwaysON = np.zeros(len(yagmur.index))
x_alwaysON[idxON] = 1
x_turnON = np.roll(x_alwaysON, 1)
x_turnON = x_alwaysON - x_turnON
x_turnON = np.where(x_turnON > 0, x_turnON, 0)

# Select Electrodes to be Used
#picked_electrodes = {'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'}
picked_electrodes = {'AF3', 'F3', 'AF4', 'F4', 'F7', 'F8'}
data_electr_filt = np.zeros((yagmur.shape[0], len(picked_electrodes)))
for n, elec in enumerate(picked_electrodes):
    data_electr_filt[:,n] = yagmur[elec]

# Epoch the Data 
N_epoch = 400 #number of samples in an epoch

epochs = np.zeros((len( np.argwhere(x_turnON) ),N_epoch, data_electr_filt.shape[1])) #34x700x4 
for n, i in enumerate(np.argwhere(x_turnON)):    
    epochs[n,:,:] = data_electr_filt[i[0]+400:i[0]+400+N_epoch,:]
    
epochs_concat = np.zeros((epochs.shape[0]*epochs.shape[2], epochs.shape[1]))
n = epochs.shape[0]
for i in range(epochs.shape[2]):
    epochs_concat[n*i:n*i+n,:] = epochs[:,:,i]


# Get the Labels
labels1 = np.zeros(len(np.argwhere(x_turnON)))
i = 0
for n in np.argwhere(x_turnON):
    labels1[i] = yagmur.label[n]
    i+=1
labels1
y = labels1

# Filter and get the CCA scores
y_est = np.zeros_like(y)
r = np.zeros([34, 3])
for i in range(34):
    X_input = np.squeeze(epochs[i,:,:])
    r[i,:] = cca_corrs(X_input, 15, 12, 10, sampling_rate)
    y_est[i] = np.argmax(r[i,:]) + 2

# Train the LDA classifier
classifier = LDA()
classifier.fit(r,y)

# Print the training accuracy
r_est = classifier.predict(r)
print(np.sum((y-r_est)==0)/34)

# Save the Model
# save the model to disk
filename = 'LDA_model.sav'
pickle.dump(classifier, open(filename, 'wb'))