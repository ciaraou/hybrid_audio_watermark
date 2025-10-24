import numpy as np
from scipy import signal

import librosa

from globals import metric_names_r, attack_names


RESAMPLE_VAL = 22050
REQUANTIZE_VAL = 8
NOISE_VAL = 20
LOWPASS_VAL = 3500
HIGHPASS_VAL = 500
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

    def detection_rate(self, tp, fn):
        """
        DR = TP / (TP + FN)
        """
        return tp / (tp + fn)

    def precision(self, tp, fp):
        """
        PR = TP / (TP + FP)
        """
        return tp / (tp + fp + 1e-10)

    def normalized_correlation(self, real, extracted):
        arr1 = np.array([int(bit) for bit in real])
        arr2 = np.array([int(bit) for bit in extracted])
        
        # Pearson correlation coefficient
        corr = np.corrcoef(arr1, arr2)[0, 1]
        
        return corr

    def bit_error_rate(self, real, diff):
        """
        BER = (Number of bit errors) / (Total number of bits)
        """
        return diff / len(real)

    def eval_all(self, real, extracted):
        if len(real) != len(extracted):
            raise ValueError("Bit strings must have the same length")
        
        diff = 0 
        tp = 0
        fp = 0
        fn = 0
        for a, b in zip(real, extracted):
            if a == '1' and b == '1':
                tp += 1
            elif a == '1':
                fn += 1
                diff += 1
            elif b == '1':
                fp += 1
                diff += 1
        # tn = sum(1 for a, b in zip(real, extracted) if a == '0' and b == '0')

        dr = self.detection_rate(tp, fn)
        pr = self.precision(tp, fp)
        nc = self.normalized_correlation(real, extracted)
        ber = self.bit_error_rate(real, diff)
        return {metric_names_r[0]: dr, metric_names_r[1]: pr, metric_names_r[2]: nc, metric_names_r[3]: ber}

    def resample(self, audio, sr):
        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.resample_val)
        return librosa.resample(audio, orig_sr=self.resample_val, target_sr=sr)

    def requantize(self, audio, sr):
        steps = 2**(self.requantize_val - 1)
        quantized = np.round(audio * steps) / steps
        return quantized
    
    def noise(self, audio, sr):
        signal_power = np.mean(audio**2)
        snr_linear = 10**(self.noise_val/10)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise
    
    def lowpass(self, audio, sr, order=2):
        nyquist = 0.5 * sr
        normalized_cutoff = self.lowpass_val / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, audio)
    
    def highpass(self, audio, sr, order=2):
        nyquist = 0.5 * sr
        normalized_cutoff = self.highpass_val / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='high')
        return signal.filtfilt(b, a, audio)
    
    def amplitude(self, audio, sr):
        return audio * self.amplitude_val
    
    def check_watermark(self, audio, wm_func, hash, wmbits):
        lsb_exhash = []
        dct_exwm = []
        if hash != "" and wmbits != "":
            hybric_func, time_func = wm_func
            lsb_exhash, _, dct_exwm = hybric_func.extract_watermark(audio, time_func, len(wmbits))
        elif wmbits != "":
            dct_exwm = wm_func.extract_watermark(audio, len(wmbits))
        elif hash != "":
            _, lsb_exhash = wm_func.extract_watermark(audio)
        else:
            print("something went wrong")
            return -1
        return self.eval_watermark(lsb_exhash, dct_exwm, hash, wmbits)
    
    def eval_watermark(self, lsb_exhash, dct_exwm, hash, wmbits):
        lsb_res = []
        dct_res = []
        if len(lsb_exhash) > 0:
            lsb_res = self.eval_all(hash, lsb_exhash)
        if len(dct_exwm) > 0:
            dct_res = self.eval_all(wmbits, dct_exwm)

        print((lsb_res, dct_res))
        return lsb_res, dct_res

    def run_suite(self, audio, sr, wm_func, hash="", wmbits="", order=2):
        """
        
        Parameters
        ----------
        audio : array_like
            watermarked audio
        sr : int
            sample rate
        wm_func : object
            WatermarkDCT, LSB, or (Hybrid, LSB)
        hash : string 
            lsb watermark (actual)
        wmbits : string
            dct/qim watermark (actual)
        
        Returns
        -------
        array_like
            3d array - out to in, 6 attacks - 2 watermarks - 4 measurements: TODO consider a struct maybe? a dict?
                resample, requantize, noise, lowpass, highpass amplitude
                lsb, dct/qim
                detection rate, precision, normalized correlation, bit error rate
        """
        if hash == "" and wmbits == "":
            print("ERROR: run_suite needs at least one watermark")
            return -1

        # attack_audio = self.resample(audio, sr)
        # rs_res = self.check_watermark(attack_audio, wm_func, hash, wmbits)
        # rs_res = rs_res[0]
        # rs_res["Watermark"] = ""
        # rs_res["Attack"] = attack_names[0]

        # attack_audio = self.requantize(audio)
        # rq_res = self.check_watermark(attack_audio, wm_func, hash, wmbits)
        # rq_res = rq_res[0]
        # rq_res["Watermark"] = ""
        # rq_res["Attack"] = attack_names[1]

        # temporarily take one watermark, TODO fix this
        picky = 0 if hash != "" else 1

        attack_audio = self.noise(audio)
        n_res = self.check_watermark(attack_audio, wm_func, hash, wmbits)
        n_res = n_res[picky] 
        n_res["Watermark"] = ""
        n_res["Attack"] = attack_names[2]
 
        attack_audio = self.lowpass(audio, sr, order)
        lp_res = self.check_watermark(attack_audio, wm_func, hash, wmbits)
        lp_res = lp_res[picky]
        lp_res["Watermark"] = ""
        lp_res["Attack"] = attack_names[3]

        attack_audio = self.highpass(audio, sr, order)
        hp_res = self.check_watermark(attack_audio, wm_func, hash, wmbits)
        hp_res = hp_res[picky]
        hp_res["Watermark"] = ""
        hp_res["Attack"] = attack_names[4]

        attack_audio = self.amplitude(audio)
        a_res = self.check_watermark(attack_audio, wm_func, hash, wmbits)
        a_res = a_res[picky]
        a_res["Watermark"] = ""
        a_res["Attack"] = attack_names[5]

        return n_res, lp_res, hp_res, a_res # rs_res, rq_res, 
    
