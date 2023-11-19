import math
from WaveUNet.model.utils import load_model
import numpy as np
import torch
import scipy
import IPython.display as ipd
import librosa
from scipy.io import wavfile as wf
from sklearn import preprocessing
import argparse
import os
from .model.utils import DataParallel, load_model
from .test import predict_song
from .model.waveunet import Waveunet

class WUN():
    def __init__(self, features=32, levels=6, feature_growth="double", 
                output_size=2.0, channels=2, instruments=["other", "vocals"], kernel_size=5, 
                depth=1, strides=4, conv_type="gn", res="fixed", separate=1):
        self.features = features
        self.levels = levels
        self.feature_growth = feature_growth
        self.output_size = output_size
        self.channels = channels
        self.instruments = instruments
        self.depth = depth
        self.strides = strides
        self.conv_type = conv_type
        self.res = res
        self.separate = separate
        self.kernel_size = kernel_size
    
    def predict(self, file_name, sr):
        num_features = [self.features*i for i in range(1, self.levels+1)] if self.feature_growth == "add" else \
                   [self.features*2**i for i in range(0, self.levels)]
        target_outputs = int(self.output_size * sr)
        model = Waveunet(self.channels, num_features, self.channels, self.instruments, kernel_size=self.kernel_size,
                        target_output_size=target_outputs, depth=self.depth, strides=self.strides,
                        conv_type=self.conv_type, res=self.res, separate=self.separate)

        cuda = False
        if torch.cuda.is_available():
            model = DataParallel(model)
            print("move model to gpu")
            model.cuda()
            cuda = True

        load_model_path = './WaveUNet/checkpoints/model'
        print("Loading model from checkpoint " + str(load_model_path))
        state = load_model(model, None, load_model_path, cuda)
        print('Step', state['step'])

        preds = predict_song(self.channels, sr, file_name, model)

        return preds['vocals']