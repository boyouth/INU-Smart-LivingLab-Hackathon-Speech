import argparse

parser = argparse.ArgumentParser()

def get_config():
    config, unparsed = parser.parse_known_args()
    return config


def str2bool(v):
    if v.lower() in ('yes', 'true', 'y', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'n', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# STEP
parser.add_argument('--step', type=int, default=0, help='Running step')

# PATH
parser.add_argument('--ctrl', type=str, default='./data/us8k_ish_train.ctl', help='Path for ctrl file')
parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset')
parser.add_argument('--feat_path', type=str, default='./data/feature', help='Path to the feature')
parser.add_argument('--save_path', type=str, default='./model/model.h5', help='Path for model')

# AUDIO
parser.add_argument('--n_fft', type=int, default=1024, help='Number of FFT components')
parser.add_argument('--sr', type=int, default=16000, help='Sampling rate of incoming signal')
parser.add_argument('--win_len', type=int, default=200, help='Number of samples between successive frames')
parser.add_argument('--hop_len', type=int, default=400, help='Each frame of audio is windowed by window()')
parser.add_argument('--mfc', type=int, default=13, help='Feature dimension of MFCC')
parser.add_argument('--filter', type=int, default=23, help='Number of Mel bands to generate for MFCC')

# MODEL
parser.add_argument('--n_class', type=int, default=10, help='Number of classes')
parser.add_argument('--n_frame', type=int, default=161, help='Number of frames')