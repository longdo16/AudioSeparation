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
    def __init__(self, device, S = 2, beta = 2, max_iter = 2000, epsilon = 1e-10, n_fft = 512, hop_length = 256):
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
        W = np.abs(np.random.normal(loc = 0, scale = 2.5, size = (K, self.S)))
        H = np.abs(np.random.normal(loc = 0, scale = 2.5, size = (self.S, N)))
        W = torch.from_numpy(W).to(self.device)
        H = torch.from_numpy(H).to(self.device)
        V = torch.from_numpy(V + self.epsilon)

        for i in range(self.max_iter):
            H = (torch.mul(H, (torch.matmul(W.T, torch.mul(torch.matmul(W, H)**(self.beta - 2), V))) / (torch.matmul(W.T, torch.matmul(W, H)**(self.beta - 1))+ self.epsilon)))
            W = (torch.mul(W, (torch.matmul(torch.mul(torch.matmul(W, H)**(self.beta - 2), V), H.T)) / (torch.matmul(torch.matmul(W, H)**(self.beta - 1), H.T)+ self.epsilon)))
        
        S1 = self.separate_source(0, W, H, angle).numpy(force = True)
        S2 = self.separate_source(1, W, H, angle).numpy(force = True)

        return S1, S2
        
    def process(self, X):
        X_stft = librosa.stft(X, n_fft = self.n_fft, hop_length = self.hop_length)
        X_stft_magnitude = np.abs(X_stft)
        X_stft_angle = np.angle(X_stft)
        V = X_stft_magnitude + self.epsilon

        return V, X_stft_angle
    
    def separate_source(self, index, W, H, angle):
        filtered_spectrograms = torch.matmul(W[:, [index]], H[[index], :])
        reconstructed_amp = filtered_spectrograms * torch.exp(1j*angle)
        reconstructed_audio = torchaudio.transforms.InverseSpectrogram(n_fft = self.n_fft)(reconstructed_amp)
        
        return reconstructed_audio