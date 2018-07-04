import numpy as np
from scipy import stats, integrate
#import scipy.fftpack
from scipy.signal import butter, lfilter, filtfilt
from scipy.linalg import inv, eig


# Filters
def butter_bandpass(lowcut, highcut, fs, order=8):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filtfilt(data, lowcut, highcut, fs, order=8):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return filtfilt(b, a, data) 


# CCA function
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

        return r_sqrt


def cca_corrs(X_input, f1, f2, f3, sampling_rate):
    'Function that gets one epoch from all electrodes: X and target frequencies: f1 f2 f3,'
    'bandpass filters the input X depending on the entered frequencies'
    'returns the correlation scores in cca space related to that frequencies'

    # BPF the input X (each electrode seperately)
    X = np.zeros_like(X_input)
    for i in range(X_input.shape[1]):
        X[:,i] = butter_bandpass_filtfilt(X_input[:,i], (np.min([f1,f2,f3])-5), (np.max([f1,f2,f3])+5), sampling_rate)
    
    # Set up the targets Y
    num_samples = X.shape[0]
    time = np.arange(0, (num_samples)/sampling_rate, 1/sampling_rate)
    
    # Target frequency and harmonics  
    Y1 = np.stack(((np.sin(2*np.pi*time*f1)  , np.cos(2*np.pi*time*f1),
                    np.sin(2*np.pi*time*f1*2), np.cos(2*np.pi*time*f1*2),
                    np.sin(2*np.pi*time*f1*4), np.cos(2*np.pi*time*f1*4),
                    np.sin(2*np.pi*time*f1*6), np.cos(2*np.pi*time*f1*6),
                   ))).T
    Y2 = np.stack(((np.sin(2*np.pi*time*f2)  , np.cos(2*np.pi*time*f2),
                    np.sin(2*np.pi*time*f2*2), np.cos(2*np.pi*time*f2*2),
                    np.sin(2*np.pi*time*f2*4), np.cos(2*np.pi*time*f2*4),
                    np.sin(2*np.pi*time*f2*6), np.cos(2*np.pi*time*f2*6),
                   ))).T
    Y3 = np.stack(((np.sin(2*np.pi*time*f3)  , np.cos(2*np.pi*time*f3),
                    np.sin(2*np.pi*time*f3*2), np.cos(2*np.pi*time*f3*2),
                    np.sin(2*np.pi*time*f3*4), np.cos(2*np.pi*time*f3*4),
                    np.sin(2*np.pi*time*f3*6), np.cos(2*np.pi*time*f3*6),
                   ))).T

    # 
    r1 = CCA_corrcoeff(X, Y1)
    r2 = CCA_corrcoeff(X, Y2)
    r3 = CCA_corrcoeff(X, Y3)

    r = np.stack(((np.max(r1), np.max(r2), np.max(r3))))
    
    return np.round(r, 7)