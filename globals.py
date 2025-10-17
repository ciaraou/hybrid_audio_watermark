from enum import Enum

metric_names = ['Detection Rate', 'Precision', 'Normalized Correlation', 'Bit Error Rate']
attack_names = ['Resample', 'Requantize', 'Noise Addition', 'Low-pass Filter', 'High-pass Filter', 'Amplitude']
watermark_names = ['Time Domain Watermark', 'Transform Domain Watermark', 'Hybrid Domain Watermark']

n_fft = 2048
hop_length = 512
smoothing_window = 10

THRESHOLD = {"lte": .01, "flatness": 0.0005, "stability": 5.0, "centroid": 3000, "zcr": 0.07}

class domain_choice(Enum):
    NONE = 0
    TIME = 1
    TRANSFORM = 2
    HYBRID = 3