import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
from scipy.io import wavfile as wf
from sklearn import preprocessing

def mix_sources(s1, s2, apply_noise = False, factor = 0.02, apply_linear_mix = False):
    max_len = max(s1.shape[0], s2.shape[0])

    if s1.shape[0] < max_len:
        s1 = process(s1, max_len)

    if s2.shape[0] < max_len:
        s2 = process(s2, max_len)

    mixture = np.c_[[s1, s2]]

    A = np.identity(2)

    if apply_linear_mix == True:
        A = np.random.rand(2,2)
        mixture = np.matmul(A, mixture)

    if apply_noise == True:
        mixture += factor * np.random.normal(size=mixture.shape)

    return mixture

def process(s, max_len):
    div = math.ceil(max_len / s.shape[0])
    new_s = np.tile(s, div)
    max_val = np.max(s)
    min_val = np.min(s)

    if max_val > 1 or min_val < 1:
        new_s = new_s / (max_val / 2) - 0.5

    new_s = 2.0 * (new_s - np.min(new_s)) / np.ptp(new_s) - 1.0

    return new_s[: max_len]

def write(X, sample_rate, dir = './', name = 'mixture'):
    wf.write(dir + name + '.wav', sample_rate, X.mean(axis=0).astype(np.float32))