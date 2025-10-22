from enum import Enum
import numpy as np

metric_names_r = ['Detection Rate', 'Precision', 'Normalized Correlation', 'Bit Error Rate']
metric_names_p = ['Objective Difference Grade', 'Distortion Index', 'Perceptible']
attack_names = ['None', 'Resample', 'Requantize', 'Noise Addition', 'Low-pass Filter', 'High-pass Filter', 'Amplitude'] # add all of them
watermark_names = ['Hybrid Domain Watermark', 'Transform Domain Watermark', 'Time Domain Watermark']

n_fft = 2048
hop_length = 1024
smoothing_window = 10


class echo_parameters:
    DELAY_0 = 0.001
    DELAY_1 = 0.005
    ALPHA = 0.2

class qim_parameters:
    ALPHA = .5
    WINDOW_FUNC = np.hanning

THRESHOLD = {"ps": [15, 35, 60], "lte": [0.05, 0.02, 0.03], "flatness": [0.001, 0.0008], "centroid": [2500, 4000]}

class domain_choice(Enum):
    NONE = 0
    TIME = 1
    TRANSFORM = 2
    HYBRID = 3

ODG_THRESHOLD = -1.0