import argparse
from ast import arg
import os
from platform import processor
from xmlrpc.client import boolean
import torch
from EnhancedAudioSeparator import EnhancedAudioSeparator

def main(device, args):
    separator = EnhancedAudioSeparator(device=device, audio_separation = args.audio_separation, speech_enhancement = args.speech_enhancement)
    if args.run_all == True:
        for entry in os.listdir(args.directory):
            separator.separate(args.directory + entry)
    else:
        separator.separate(args.file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_separation', type=str, default='WUN',
                        help="Audio Separation Model")
    parser.add_argument('--speech_enhancement', type=str, default='SS',
                        help="Speech Enhancement Method")
    parser.add_argument('--file', type=str, default='./data/mixture.wav',
                        help="Input File")
    parser.add_argument('--run_all', type=bool, default=True,
                        help='Run All Files in a Directory')
    parser.add_argument('--directory', tyoe=str, default='./data/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    main(device, args)
