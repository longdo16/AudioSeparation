import math
import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
from scipy.io import wavfile as wf
from sklearn import preprocessing

# Implement ICA for audio source separation here

class ICA():
    def __init__(self, device, lr = 0.01, max_iter = 2000, factor = 100.0, epsilon = 1e-10, l = 10000):
        self.device = device
        self.lr = lr
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.factor = factor
        self.l = l
    
    def predict_batch(self, X):
        n = X.shape[0] # Number of source signal
        t = X.shape[1] # Time
        m = X.shape[0]

        # Initialize the (n by m) matrix W with small random values
        W = np.random.rand(n, m) / 10.0 + 1e-10

        for i in range(10000):

          temp = t // self.l

          sumDeltaW = np.zeros((n, m))

          for j in range(temp + 1):
            data = X[:, j * 1 : (j + 1) * 1]

            # Calculate Y = WX (Y is our current estimate of the source signals)
            Y = np.matmul(W, data)

            # Calculate Z
            Z = 1.0 / (1.0 + np.exp(-1 * Y))

            # Find delta W
            deltaW = self.lr * np.matmul((np.identity(n) + np.matmul((1.0 - 2.0 * Z), Y.T)), W)

            sumDeltaW += deltaW

            # deltaW = deltaW / np.linalg.norm(deltaW)

            # Update W
            # W = W + deltaW
          W = W + sumDeltaW

          if i != 0 and i % 100 == 0:
            print(np.linalg.norm(sumDeltaW))

        S = np.matmul(W, X)
        return S[0, :], S[1, :]
    
    def predict(self, X):
        n = X.shape[0]
        t = X.shape[1]
        m = X.shape[0]
        W = np.random.rand(n, m) / self.factor + self.epsilon
        W = torch.from_numpy(W).to(self.device)
        X = torch.from_numpy(X).to(self.device)

        for i in range(self.max_iter):
            # Current estimate of the source signal
            Y = torch.matmul(W, X)

            Z = 1.0 / (1.0 + torch.exp(-1.0 * Y))
            I = torch.eye(n).to(self.device)
            delta_W = self.lr * torch.matmul((I + torch.matmul((1.0 - 2.0 * Z), Y.T)), W)
            W = W + delta_W

        S = torch.matmul(W, X).numpy(force = True)
        return S[0, :], S[1, :]