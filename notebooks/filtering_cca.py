import numpy as np
from scipy import stats, integrate

from scipy.signal import butter, lfilter, filtfilt
from scipy.linalg import inv, eig, eigh

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel


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


# Create Harmonics
def get_harmonics(f, time):
    Y = np.stack(((np.sin(2*np.pi*time*f)  , np.cos(2*np.pi*time*f),
                   np.sin(2*np.pi*time*f*2), np.cos(2*np.pi*time*f*2),
                   np.sin(2*np.pi*time*f*4), np.cos(2*np.pi*time*f*4),
                   np.sin(2*np.pi*time*f*6), np.cos(2*np.pi*time*f*6),
                   ))).T
    return Y


# CCA function

def CCA_corrcoeff(X, Y, reg=1):
        """Function calculates correlations coeffciencts r"""
        Z = np.column_stack((X, Y,))
        C = np.cov(Z.T)

        sy = np.shape(Y)[1] if Y.ndim > 1 else 1
        sx = np.shape(X)[1] if X.ndim > 1 else 1

        Cxx = C[0:sx, 0:sx] + 10**(-8) * np.eye(sx) * reg
        Cxy = C[0:sx, sx:sx+sy]
        Cyx = Cxy.T
        Cyy = C[sx:sx+sy, sx:sx+sy] + 10**(-8) * np.eye(sy) * reg
        invCyy = inv(Cyy)
        invCxx = inv(Cxx)

        r, Wx = eig( np.dot( np.dot(invCxx, Cxy), np.dot(invCyy, Cyx) ) )
        r = np.real(r)
        r_sqrt = np.sqrt(np.round(r, 7))

        return r_sqrt


## Kernel CCA functions

def _make_kernel(d, normalize=True, ktype='linear', gausigma=1.0, degree=2):
    """Makes a kernel for data d
      If ktype is 'linear', the kernel is a linear inner product
      If ktype is 'gaussian', the kernel is a Gaussian kernel, sigma = gausigma
      If ktype is 'poly', the kernel is a polynomial kernel with degree=degree
    """
    d = np.nan_to_num(d)
    cd = d - d.mean(0)

    if ktype == 'linear':
        kernel = np.dot(cd, cd.T)
    elif ktype == 'gaussian':
        pairwise_dists = squareform(pdist(d, 'euclidean'))
        kernel = np.exp(-pairwise_dists ** 2 / 2 * gausigma ** 2)
    elif ktype == 'poly':
        kernel = np.dot(cd, cd.T) ** degree
    elif ktype == 'rbf':
        kernel = rbf_kernel(cd)

    kernel = (kernel + kernel.T) / 2.
    if normalize:
        kernel = kernel / np.linalg.eigvalsh(kernel).max()

    return kernel

def _make_kernel_gaussian(d, normalize=True, gausigma=1.0, degree=2):
    """Makes a kernel for data d
      If ktype is 'linear', the kernel is a linear inner product
      If ktype is 'gaussian', the kernel is a Gaussian kernel, sigma = gausigma
      If ktype is 'poly', the kernel is a polynomial kernel with degree=degree
    """
    d = np.nan_to_num(d)
    cd = d - d.mean(0)

    pairwise_dists = squareform(pdist(d, 'euclidean'))
    kernel = np.exp(-pairwise_dists ** 2 / 2 * gausigma ** 2)

    kernel = (kernel + kernel.T) / 2.
    if normalize:
        kernel = kernel / np.linalg.eigvalsh(kernel).max()

    return kernel

def kcca(X, Y, reg=0., numCC=None, ktype='linear', gausigma=1.0, degree=2):
    """Set up and solve the kernel CCA eigenproblem
    """
    X = _make_kernel(X, ktype=ktype, gausigma=gausigma, degree=degree)
    Y = _make_kernel(Y, ktype=ktype, gausigma=gausigma, degree=degree)

    r = CCA_corrcoeff(X, Y, reg)
    return r

def kcca_gaussian(X, Y, reg=0., numCC=None,  gausigma=1.0, degree=2):
    """Set up and solve the kernel CCA eigenproblem
    """
    X = _make_kernel_gaussian(X, gausigma=gausigma, degree=degree)
    Y = _make_kernel_gaussian(Y, gausigma=gausigma, degree=degree)

    r = CCA_corrcoeff(X, Y, reg)
    return r


# Final Function
def cca_corrs(X_input, f1, f2, f3, sampling_rate):
    '''
    Function that gets one epoch from all electrodes: X and target frequencies: f1 f2 f3,
    bandpass filters the input X depending on the entered frequencies
    returns the correlation scores in cca space related to that frequencies
    '''

    # BPF the input X (each electrode seperately)
    X = np.zeros_like(X_input)
    for i in range(X_input.shape[1]):
        X[:, i] = butter_bandpass_filtfilt(X_input[:, i], (np.min([f1, f2, f3])-5), (np.max([f1, f2, f3])+5), sampling_rate)

    # Set up the targets Y
    num_samples = X.shape[0]
    time = np.arange(0, (num_samples)/sampling_rate, 1/sampling_rate)

    # Set up harmonics
    Y1 = get_harmonics(f1, time)
    Y2 = get_harmonics(f2, time)
    Y3 = get_harmonics(f3, time)

    # Do CCA
    r1 = CCA_corrcoeff(X, Y1)
    r2 = CCA_corrcoeff(X, Y2)
    r3 = CCA_corrcoeff(X, Y3)

    r = np.stack(((np.max(r1), np.max(r2), np.max(r3))))

    return np.round(r, 7)

# Final Function
def kcca_corrs(X_input, f1, f2, f3, sampling_rate, ktype='linear', reg=1, gausigma=1.0, degree=2):
    '''
    Function that gets one epoch from all electrodes: X and target frequencies: f1 f2 f3,
    bandpass filters the input X depending on the entered frequencies
    returns the correlation scores in cca space related to that frequencies
    '''

    # BPF the input X (each electrode seperately)
    X = np.zeros_like(X_input)
    for i in range(X_input.shape[1]):
        X[:, i] = butter_bandpass_filtfilt(X_input[:, i], (np.min([f1, f2, f3])-5), (np.max([f1, f2, f3])+5), sampling_rate)

    # Set up the targets Y
    num_samples = X.shape[0]
    time = np.arange(0, (num_samples)/sampling_rate, 1/sampling_rate)

    # Set up harmonics
    Y1 = get_harmonics(f1, time)
    Y2 = get_harmonics(f2, time)
    Y3 = get_harmonics(f3, time)

    # Do CCA
    r1 = kcca_gaussian(X, Y1, reg=reg, gausigma=gausigma, degree=degree)
    r2 = kcca_gaussian(X, Y2, reg=reg, gausigma=gausigma, degree=degree)
    r3 = kcca_gaussian(X, Y3, reg=reg, gausigma=gausigma, degree=degree)


    r = np.stack(((np.max(r1), np.max(r2), np.max(r3))))

    return np.round(r, 7)
