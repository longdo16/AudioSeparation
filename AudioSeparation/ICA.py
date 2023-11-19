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
    def __init__(self, device, lr = 0.0001, max_iter = 10000, factor = 10.0, epsilon = 1e-10, l = 100000):
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
        W = np.random.rand(n, m) / self.factor + self.epsilon

        for i in range(self.max_iter):

          temp = t // self.l
          sumDeltaW = np.zeros((n, m))

          for j in range(temp + 1):

            data = X[:, j * 1 : (j + 1) * 1]
            Y = np.matmul(W, data)
            Z = 1.0 / (1.0 + np.exp(-1 * Y))
            deltaW = self.lr * np.matmul((np.identity(n) + np.matmul((1.0 - 2.0 * Z), Y.T)), W)
            sumDeltaW += deltaW

          W = W + sumDeltaW

        S = np.matmul(W, X)

        return S[0, :], S[1, :]
    
    def predict(self, X):
        n = X.shape[0]
        t = X.shape[1]
        m = X.shape[0]
        W = np.random.rand(n, m) / self.factor + self.epsilon

        for i in range(self.max_iter):
            # Current estimate of the source signal
            Y = np.matmul(W, X)

            Z = 1.0 / (1.0 + np.exp(-1.0 * Y))
            delta_W = self.lr * np.matmul((np.identity(n) + np.matmul((1.0 - 2.0 * Z), Y.T)), W)
            W = W + delta_W

        S = np.matmul(W, X)
        return S[0, :], S[1, :]