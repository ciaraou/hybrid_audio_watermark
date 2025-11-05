import numpy as np
from scipy import signal

import librosa

from globals import metric_names_r, attack_names


RESAMPLE_VAL = 22050
REQUANTIZE_VAL = 8
NOISE_VAL = 20
LOWPASS_VAL = 6000
HIGHPASS_VAL = 300
AMPLITUDE_VAL = 1.1
# TODO:
# compression
# echo
# desync


class RobustnessEvaluation():
    """
    Checks robustness against a variety of "attacks":
        Re-sampling
        Re-quantization
        Noise corruption
        Low-pass filtering
        High-pass filtering
        File compression (TODO)
        Echo addition (TODO)
        Amplitude scaling
        Desynchronization (TODO)
    
    """
    def __init__(self):
        self.resample_val = RESAMPLE_VAL
        self.requantize_val = REQUANTIZE_VAL
        self.noise_val = NOISE_VAL
        self.lowpass_val = LOWPASS_VAL
        self.highpass_val = HIGHPASS_VAL
        self.amplitude_val = AMPLITUDE_VAL

    # --- metrics ---

    def detection_rate(self, *, tp: int, fn: int) -> float:
        """
        DR = TP / (TP + FN)
        """
        return tp / (tp + fn)

    def precision(self, *, tp: int, fp: int) -> float:
        """
        PR = TP / (TP + FP)
        """
        return tp / (tp + fp + 1e-10)

    def normalized_correlation(self, *, real: str, extracted: str) -> float:
        arr1 = np.array([int(bit) for bit in real])
        arr2 = np.array([int(bit) for bit in extracted])
        
        # Pearson correlation coefficient
        corr = np.corrcoef(arr1, arr2)[0, 1]
        
        return corr

    def bit_error_rate(self, *, real: str, diff: int) -> float:
        """
        BER = (Number of bit errors) / (Total number of bits)
        """
        return diff / len(real)
    
    # --- eval helpers ---
    def recover_extracted(self, *, extracted: str, flagbits: str) -> str:
        """
        Reconstruct the flag from extracted redundant flag repetitions. 
        Check each index of the watermark for any accurate bit in the extracted string.
        ex: extracted=11000, real=101. possible watermarks are 110 and 00. recovered=100
        
        Param:
        extracted : str 
            extracted repeated watermark in bit string form
        flagbits: str
            original watermark in bit string form

        Returns:
        str
            recovered watermark in its most accurate form
        """
        flag_len = len(flagbits)

        recovered = list(extracted[:flag_len])
        for i in range(flag_len):
            curr = i
            while(curr < len(extracted) and recovered[i] != flagbits[i]):
                if extracted[curr] == flagbits[i]:
                    recovered[i] = flagbits[i]
                    break
                curr += flag_len
        return ''.join(recovered)
        
    # --- eval all metrics ---
    def eval_all(self, *, extracted: str, flag: list[int]) -> dict[str, float]:
        """
        Runs all metrics on extracted watermark bits against original flag bits
        
        Params:
        extracted : str 
            extracted repeated watermark in bit string form
        flag: list[int]
            original watermark in list of int form. bits 1 or 0.
        
        Returns:
        dict{str: float}
            Results according to metric_names_r
        """
        # flagbits = ("".join(str(bit) for bit in flag)* ((len(extracted) + len(flag)) // len(flag)))[:len(extracted)]
        # extractedflag = extracted
        flagbits = "".join(str(bit) for bit in flag)
        extractedflag = self.recover_extracted(extracted=extracted, flagbits=flagbits)
        
        diff = 0 
        tp = 0
        fp = 0
        fn = 0
        for a, b in zip(flagbits, extractedflag):
            if a == '1' and b == '1':
                # true positive: extracted a 1 (b) where there is a 1 (a)
                tp += 1
            elif a == '1':
                # false negative: extracted a 0 (b) where there is a 1 (a)
                fn += 1
                diff += 1
            elif b == '1':
                # false positive: extracted a 1 (b) where there is a 0 (a)
                fp += 1
                diff += 1
            # true negative not used

        dr = self.detection_rate(tp=tp, fn=fn)
        pr = self.precision(tp=tp, fp=fp)
        nc = self.normalized_correlation(real=flagbits, extracted=extractedflag)
        ber = self.bit_error_rate(real=flagbits, diff=diff)
        return {metric_names_r[0]: dr, metric_names_r[1]: pr, metric_names_r[2]: nc, metric_names_r[3]: ber}

    # --- audio transformation "attacks" ---
    def resample(self, *, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.resample_val)
        return librosa.resample(audio, orig_sr=self.resample_val, target_sr=sr)

    def requantize(self, *, audio: np.ndarray, sr: int) -> np.ndarray:
        steps = 2**(self.requantize_val - 1)
        quantized = np.round(audio * steps) / steps
        return quantized
    
    def noise(self, *, audio: np.ndarray, sr: int) -> np.ndarray:
        signal_power = np.mean(audio**2)
        snr_linear = 10**(self.noise_val/10)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise
    
    def lowpass(self, *, audio: np.ndarray, sr: int, order: int = 2) -> np.ndarray:
        nyquist = 0.5 * sr
        normalized_cutoff = self.lowpass_val / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, audio)
    
    def highpass(self, *, audio: np.ndarray, sr: int, order: int = 2) -> np.ndarray:
        nyquist = 0.5 * sr
        normalized_cutoff = self.highpass_val / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='high')
        return signal.filtfilt(b, a, audio)
    
    def amplitude(self, *, audio: np.ndarray, sr: int) -> np.ndarray:
        return audio * self.amplitude_val
    