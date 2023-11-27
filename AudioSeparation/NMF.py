import math
import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display
import IPython.display as ipd
from scipy.io import wavfile as wf
import torchaudio

# Implement NMF for audio source separation here

class NMF():
    def __init__(self, device, S = 2, beta = 2, max_iter = 200, epsilon = 1e-10, n_fft = 512, hop_length = 256):
        self.device = device
        self.S = S
        self.beta = beta
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.n_fft = n_fft
        self.hop_length = hop_length

    def predict(self, X):
        V, angle = self.process(X)
        K = V.shape[0]
        N = V.shape[1]
        W = np.abs(np.random.normal(loc=0, scale = 2.5, size=(K, self.S)))
        H = np.abs(np.random.normal(loc=0, scale = 2.5, size=(self.S, N)))

        for i in range(self.max_iter):

            if i % 10 == 0:
              print(i // 10)
            
            H = (np.multiply(H, (np.matmul(W.T, np.multiply(np.matmul(W, H)**(self.beta - 2), V))) / (np.matmul(W.T, np.matmul(W, H)**(self.beta - 1))+ 10e-10)))
            W = (np.multiply(W, (np.matmul(np.multiply(np.matmul(W, H)**(self.beta - 2), V), H.T)) / (np.matmul(np.matmul(W, H)**(self.beta - 1), H.T)+ 10e-10)))
        
        S1 = self.separate_source(0, W, H, angle)
        S2 = self.separate_source(1, W, H, angle)

        return S1, S2
        
    def process(self, X):
        X_stft = librosa.stft(X, n_fft = self.n_fft, hop_length = self.hop_length)
        X_stft_magnitude = np.abs(X_stft)
        X_stft_angle = np.angle(X_stft)
        V = X_stft_magnitude + self.epsilon

        return V, X_stft_angle
    
    def separate_source(self, index, W, H, angle):
        filtered_spectrograms = np.matmul(W[:,[index]], H[[index],:])
        reconstructed_amp = filtered_spectrograms[index] * np.exp(1j * angle)
        reconstructed_audio = librosa.istft(reconstructed_amp, n_fft = self.n_fft, hop_length = self.hop_length)
        
        return reconstructed_audio