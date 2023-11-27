from ctypes import util
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
from scipy.io import wavfile as wf
from sklearn import preprocessing
from utils import mix_sources
from utils import process
from utils import write
from utils import spectral_substraction
from utils import wiener_filtering
from AudioSeparation.ICA import ICA
from AudioSeparation.NMF import NMF
from WaveUNet.WUN import WUN
from SpeechEnhancement.SS import SS
import torch
import os

class EnhancedAudioSeparator():
    def __init__(self, device, audio_separation = 'ICA', speech_enhancement = 'SS', denoise_later = True, denoise_before = False):
        self.device = device
        self.model = ICA(device, lr = 0.0001, max_iter = 1000, l = 10000, factor = 10.0)
        self.audio_separation = audio_separation
        self.speech_enhancement = speech_enhancement
        self.denoise_later = denoise_later
        self.denoise_before = denoise_before
        self.dir = './ICA/'

        if self.audio_separation == 'NMF':
            self.model = NMF(device)
            self.dir = './NMF/'
        elif self.audio_separation == 'WUN':
            self.model = WUN()
            self.dir = './WUN/'
    
    def separate(self, file):
        s, sr = librosa.load(file)

        if self.denoise_before == True:
            if self.speech_enhancement == 'SS':
                s = spectral_substraction(s)
            elif self.speech_enhancement == 'WF':
                s = wiener_filtering(s)

        separated_s1 = None
        separated_s2 = None
        file_name = os.path.basename('mixture.wav')[:-4]
        
        if self.audio_separation == 'ICA':
            X = np.c_[[s, s]]
            separated_s1, separated_s2 = self.model.predict_batch(X)
        elif self.audio_separation == 'NMF':
            X = s
            separated_s1, separated_s2 = self.model.predict(X)
        else:
            X = s
            X_1 = X[: X.shape[0] // 2,]
            X_2 = X[X.shape[0] // 2:, ]
            wf.write('./mixture/' + str(file_name) + '_part_1.wav', sr, X_1.astype(np.float32))
            wf.write('./mixture/' + str(file_name) + '_part_2.wav', sr, X_2.astype(np.float32))
            separated_s1 = self.model.predict('./mixture/' + str(file_name) + '_part_1.wav', sr)
            separated_s2 = self.model.predict('./mixture/' + str(file_name) + '_part_2.wav', sr)
            separated_s1 = np.squeeze(separated_s1.T)
            separated_s2 = np.squeeze(separated_s2.T)

        if self.denoise_later == True:
            if self.speech_enhancement == 'SS':
                separated_s1 = spectral_substraction(separated_s1)
                separated_s2 = spectral_substraction(separated_s2)
            elif self.speech_enhancement == 'WF':
                separated_s1 = wiener_filtering(separated_s1)
                separated_s2 = wiener_filtering(separated_s2)
        
        if self.audio_separation == 'WUN':
            X = np.concatenate((separated_s1, separated_s2))
            wf.write(self.dir + str(file_name) + '_separated.wav', sr, X)
        else:
            wf.write(self.dir + str(file_name) + '_separated_s1.wav', sr, separated_s1)
            wf.write(self.dir + str(file_name) +'_separated_s2.wav', sr, separated_s2)
