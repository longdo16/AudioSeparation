import argparse
from ast import arg
import os
from platform import processor
import torch
from Processor import Processor

def main(device, args):
    processor = Processor(device = device, audio_separation = args.audio_separation, speech_enhancement = args.speech_enhancement)
    processor.process(args.file_1, args.file_2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_separation', type=str, default='ICA',
                        help="Audio Separation Model")
    parser.add_argument('--speech_enhancement', type=str, default='SS',
                        help="Speech Enhancement Method")
    parser.add_argument('--file_1', type=str, default='./data/talkonly.wav',
                        help="Speaker File")
    parser.add_argument('--file_2', type=str, default='./data/piano.wav',
                        help="Sound or Noise File")                
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    main(device, args)