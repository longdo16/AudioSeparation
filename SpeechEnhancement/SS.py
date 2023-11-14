import math
import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
from scipy.io import wavfile as wf
from sklearn import preprocessing

# Implement Spectral Subtraction for speech enhancement here

class SS():
    def __init__(self, n_fft = 512, hop_length = 256, N = 10, a = 25, beta = 0.002) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.N = N
        self.a = a
        self.beta = beta
    
    def spectral_oversubtraction(self, X):
        Y, est_Mn, est_Pn = self.noise_estimation_snr(X)
        snr = 10 * np.log10(np.sum(np.abs(Y) ** 2) / sum(est_Pn))
        alpha = list()

        for gamma in snr:

            if -5 <= gamma <= 20:

                alpha.append(-6.25 * gamma / 25 + 6)

            elif gamma > 20:

                alpha.append(1)

            else:

                alpha.append(7.25)

        est_powX = np.maximum(np.abs(Y) ** 2 - alpha * est_Pn, self.beta * est_Pn)
        angle = np.angle(Y)
        temp = np.sqrt(est_powX) * np.exp(1j * angle)
        result = librosa.istft(temp.T, n_fft = self.n_fft, hop_length = self.hop_length)

        return result


    def noise_estimation_snr(self, X):
        Y = librosa.stft(X, n_fft = self.n_fft, hop_length = self.hop_length)
        est_Mn = np.zeros(Y.shape[0])
        est_Pn = np.zeros(Y.shape[0])

        for m in range(Y.shape[0]):

            if m < self.N:

                est_Mn = np.abs(Y[m])
                est_Pn = np.abs(Y[m]) ** 2

            else:

                sigmoid = np.abs(Y[m]) ** 2 / np.mean(np.abs(Y[m - self.N: m]) ** 2, axis = 0)
                alpha = 1.0 / (1 + np.exp(-self.a * (sigmoid - 1.5)))
                est_Mn[m] = alpha * np.abs(est_Mn[m - 1]) + (1 - alpha) * np.abs(Y[m])
                est_Pn[m] = alpha * (np.abs(est_Mn[m - 1]) ** 2) + (1 - alpha) * (np.abs(Y[m]) ** 2)

        return Y, est_Mn, est_Pn
    
    def spectral_subtraction_mag(self, X):
        Y, est_Mn, _ = self.noise_estimation_snr(X)
        est_magX = np.maximum(np.abs(Y) - est_Mn, 0)
        angle = np.angle(Y)

        temp = est_magX * np.exp(1j * angle)
        result = librosa.istft(temp.T, n_fft = self.n_fft, hop_length = self.hop_length)

        return result
    
    def spectral_subtraction_pow(self, X):
        Y, est_Mn, est_Pn = self.noise_estimation_snr(X)
        est_powX = np.maximum(np.abs(Y) ** 2 - est_Pn, 0)
        angle = np.angle(Y)

        temp = est_powX * np.exp(1j * angle)
        result = librosa.istft(temp.T, n_fft = self.n_fft, hop_length = self.hop_length)

        return result