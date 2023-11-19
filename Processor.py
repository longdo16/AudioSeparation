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

class Processor():
    def __init__(self, device, audio_separation = 'ICA', speech_enhancement = 'SS', denoise_later = True, denoise_before = False):
        self.device = device
        self.model = ICA(device, lr = 0.0001, max_iter = 1000, l = 10000, factor = 10.0)
        self.audio_separation = audio_separation
        self.speech_enhancement = speech_enhancement
        self.denoise_later = denoise_later
        self.denoise_before = denoise_before
        self.dir = './ICA/'

        if self.audio_separation == 'NMF':
            self.model = NMF()
            self.dir = './NMF/'
        elif self.audio_separation == 'WUN':
            self.model = WUN()
            self.dir = './WUN/'
            pass
    
    def process(self, s1_file, s2_file, apply_noise = False, factor = 0.02, apply_linear_mix = True):
        # assume s1 is the speaker wav file
        s1, sr1 = librosa.load(s1_file)
        s2, sr2 = librosa.load(s2_file, sr = sr1)
        sample_rate = sr1

        if self.denoise_before == True:
            if self.speech_enhancement == 'SS':
                s1 = spectral_substraction(s1)
                s2 = spectral_substraction(s2)
            elif self.speech_enhancement == 'WF' :
                s1 = wiener_filtering(s1)
                s2 = wiener_filtering(s2)

        X = mix_sources(s1, s2, apply_noise = apply_noise, factor = factor, apply_linear_mix = apply_linear_mix)
        wf.write('./mixture/talk_and_music.wav', sample_rate, X.mean(axis=0).astype(np.float32))

        if self.audio_separation == 'ICA':
            separated_s1, separated_s2 = self.model.predict_batch(X)
        elif self.audio_separation == 'NMF':
            X, sr = librosa.load('./mixture/talk_and_music.wav', sr = sample_rate)
            separated_s1, separated_s2 = self.model.predict(X)
        else:
            X = X.mean(axis=0).astype(np.float32)
            X_1 = X[: X.shape[0] // 2,]
            X_2 = X[X.shape[0] // 2:, ]
            wf.write('./mixture/talk_and_music_1.wav', sample_rate, X_1.astype(np.float32))
            wf.write('./mixture/talk_and_music_2.wav', sample_rate, X_2.astype(np.float32))
            separated_s1 = self.model.predict('./mixture/talk_and_music_1.wav', sample_rate)
            separated_s2 = self.model.predict('./mixture/talk_and_music_2.wav', sample_rate)
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
            wf.write(self.dir + 'separated.wav', sample_rate, X)
        else:
            wf.write(self.dir + 'separated_s1.wav', sample_rate, separated_s1)
            wf.write(self.dir + 'separated_s2.wav', sample_rate, separated_s2)