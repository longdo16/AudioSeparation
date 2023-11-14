import imp
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
from AudioSeparation.ICA import ICA
from AudioSeparation.NMF import NMF
import torch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    s1_file = './data/piano.wav'
    s2_file = './data/talkonly.wav'
    sample_rate = 22050

    s1, sr = librosa.load(s1_file, sr = sample_rate)

    s2, sr = librosa.load(s2_file, sr = sample_rate)

    X = mix_sources(s1, s2, False, 0.02, True)
    wf.write('./talk_and_music.wav', sample_rate, X.mean(axis=0).astype(np.float32))
    
    ica_model = ICA(device, lr = 0.0001, max_iter = 1000, l = 10000)
    nmf_model = NMF(device)

    X = mix_sources(s1, s2, False, 0.02, True)

    separated_s1, separated_s2 = ica_model.predict_batch(X)

    wf.write('./ICAseparated_s1.wav', sample_rate, separated_s1)
    wf.write('./ICAseparated_s2.wav', sample_rate, separated_s2)

if __name__ == '__main__':
    main()