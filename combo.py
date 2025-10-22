import librosa 
from globals import n_fft, hop_length, smoothing_window, THRESHOLD, domain_choice, echo_parameters, qim_parameters
import numpy as np

import utils.watermark.util as watermark_utils
from utils.watermark.dct_transform import DCT
from utils.watermark.window import Window

# flatness
def domain_flatness(y, sr):
    return librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]

# harmonicity 
def domain_pitch_stability(y, sr):
    # Pitch estimation (f0) with librosa.pyin
    f0, _, _ = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'), 
        sr=sr, 
        frame_length=n_fft, 
        hop_length=hop_length
    )

    # Pitch stability: std dev of f0 over sliding windows (ignore NaNs)
    stability = []
    for i in range(len(f0)):
        start = max(0, i - smoothing_window + 1)
        window_f0 = f0[start:i+1]
        window_f0 = window_f0[~np.isnan(window_f0)]
        if len(window_f0) > 1:
            stability.append(np.std(window_f0))
        else:
            stability.append(np.nan)
    return np.array(stability)


def domain_long_term_energy(y, sr):
    # RMS energy per frame
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]

    # Long-term energy average (moving average)
    window = np.ones(smoothing_window) / smoothing_window
    smoothed = np.convolve(rms, window, mode='same')
    return smoothed 

def domain_zcr(y, sr):
    return librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]

def domain_centroid(y, sr):
    return librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]

def determine_domain(lte, flatness, ps, zcr, centroid):
    # prioritize LTE and pitch stability, TODO. second: centroid and flatnes
    # if np.isnan(ps):
    #     if lte > THRESHOLD["lte"][0]:
    #         return domain_choice.HYBRID 
    #     else:
    #         return domain_choice.TIME 
    # elif ps < THRESHOLD["ps"][0]:
    #     return domain_choice.TRANSFORM
    # elif ps < THRESHOLD["ps"][1]:
    #     if centroid > THRESHOLD["centroid"][0] and lte > THRESHOLD["lte"][1]:
    #         return domain_choice.HYBRID 
    #     elif centroid > THRESHOLD["centroid"][1]:
    #         return domain_choice.TIME
    #     elif flatness > THRESHOLD["flatness"][0]:
    #         return domain_choice.HYBRID 
    #     else:
    #         return domain_choice.TRANSFORM
    # elif ps < THRESHOLD["ps"][2]:
    #     if lte > THRESHOLD["lte"][2] and flatness < THRESHOLD["flatness"][1]:
    #         return domain_choice.HYBRID
    #     else:
    #         return domain_choice.TIME
    # else:
    #     return domain_choice.TIME
    if np.isnan(ps):
        # if lte > THRESHOLD["lte"][0]:
        #     return domain_choice.HYBRID 
        # else:
        return domain_choice.TIME 
    elif ps < THRESHOLD["ps"][0]:
        return domain_choice.TRANSFORM
    elif ps < THRESHOLD["ps"][1]:
        # if centroid > THRESHOLD["centroid"][0] and lte > THRESHOLD["lte"][1]:
        #     return domain_choice.HYBRID 
        if centroid > THRESHOLD["centroid"][1]:
            return domain_choice.TIME
        # elif flatness > THRESHOLD["flatness"][0]:
        #     return domain_choice.HYBRID 
        else:
            return domain_choice.TRANSFORM
    elif ps < THRESHOLD["ps"][2]:
        # if lte > THRESHOLD["lte"][2] and flatness < THRESHOLD["flatness"][1]:
        #     return domain_choice.HYBRID
        # else:
        return domain_choice.TIME
    else:
        return domain_choice.TIME
    # if lte < THRESHOLD["lte"]:
    #     return domain_choice.NONE
    
    # if flatness > THRESHOLD["flatness"] and (zcr > THRESHOLD["zcr"] or centroid > THRESHOLD["centroid"]):
    #     return domain_choice.TIME 
    
    # if flatness < THRESHOLD["flatness"] and ps < THRESHOLD["stability"]:
    #     return domain_choice.TRANSFORM
    
    # return domain_choice.HYBRID

def precalculate_frame_domains(audio, sr):
    flatness = domain_flatness(audio, sr)
    lte = domain_long_term_energy(audio, sr)
    ps = domain_pitch_stability(audio, sr)
    zcr = domain_zcr(audio, sr)
    centroid = domain_centroid(audio, sr)

    min_len = min([len(arr) for arr in [flatness, lte, ps, zcr, centroid]])

    domain_list = []

    for i in range(min_len):
        domain_list.append(determine_domain(lte[i], flatness[i], ps[i], zcr[i], centroid[i]))

    return domain_list


class Echo():
    
    def embed_watermark_bit(self, *, frame, sr, wmbit):
        """
        
        Parameters
        ----------
        x : array_like
            1d host audio array
        """

        delay = echo_parameters.DELAY_0 if wmbit == 0 else echo_parameters.DELAY_1
        sample = int(delay*sr)

        res = frame.copy()

        res[sample%len(res):] += echo_parameters.ALPHA*frame[:-sample%len(res)]

        # normalize
        max_val = np.max(np.abs(res))
        if max_val > 1.0:
            res = res / max_val

        return res
    
    def full_embed_watermark(self, y, sr, wmbits):
        frames = len(y)//hop_length

        for i in range(frames):
            y[i*hop_length:(i+1)*hop_length] = self.embed_watermark_bit(frame=y[i*hop_length:(i+1)*hop_length], sr=sr, wmbit=wmbits[i%len(wmbits)])
    
        return y
    
    def extract_watermark_bit(self, *, frame, sr):
        # Convert delays from seconds to samples
        delay_samples_0 = int(echo_parameters.DELAY_0 * sr)
        delay_samples_1 = int(echo_parameters.DELAY_1 * sr)
        
        # Compute the cepstrum
        # 1. Take FFT of the signal
        spectrum = np.fft.fft(frame)
        
        # 2. Take log magnitude (add small epsilon to avoid log(0))
        log_spectrum = np.log(np.abs(spectrum) + 1e-10)
        
        # 3. Take inverse FFT to get cepstrum
        cepstrum = np.real(np.fft.ifft(log_spectrum))
        
        # 4. Look at the magnitude of cepstrum at both delay positions
        # The cepstrum will have a peak at the echo delay (in samples, called "quefrency")
        if delay_samples_0 < len(cepstrum):
            cepstrum_0 = np.abs(cepstrum[delay_samples_0])
        else:
            cepstrum_0 = 0
        
        if delay_samples_1 < len(cepstrum):
            cepstrum_1 = np.abs(cepstrum[delay_samples_1])
        else:
            cepstrum_1 = 0
        
        # Optional: look at a small window around each delay for more robustness
        window = 2
        if delay_samples_0 + window < len(cepstrum):
            cepstrum_0 = np.max(np.abs(cepstrum[max(0, delay_samples_0-window):delay_samples_0+window+1]))
        
        if delay_samples_1 + window < len(cepstrum):
            cepstrum_1 = np.max(np.abs(cepstrum[max(0, delay_samples_1-window):delay_samples_1+window+1]))
        
        # The bit corresponds to whichever delay has stronger cepstral peak
        extracted_bit = 0 if cepstrum_0 > cepstrum_1 else 1
        
        return extracted_bit
    
    def full_extract_watermark(self, y, sr):
        frames = len(y)//hop_length

        res = []
        for i in range(frames):
            b = self.extract_watermark_bit(frame=y[i*hop_length:(i+1)*hop_length], sr=sr)
            res.append(str(b))
        return res

class WatermarkDCT(DCT):
    """
    (M)DCT transformation domain watermark 
    """

    def __init__(self, window_func, watermark): # maybe make watermark optional, random gen? TODO add win_type option if needed
        """
        Initializes MDCT object with window

        Parameters
        ----------
        window : function
            windowing function for MDCT super class initialization
        watermark : bit string 
            watermark
        """
        self.watermark = watermark
        self.N = len(watermark)
        self.window = Window(self.N, window_func)
        super().__init__(self.window)

    def frequency_bands(self, res_N_fb):
        return res_N_fb // 4, res_N_fb // 2
    
    def get_coef_i(self, bins0_gci, i_gci, coefs_per_frame_gci):
        return bins0_gci + (i_gci % coefs_per_frame_gci)


    def _embed_qim_bit_helper(self, frame_coeffs, bit, coef_i, alpha):
        """
        Embed a single bit into one MDCT frame using QIM. This takes pre-windowed frames
        
        Parameters:
        -----------
        frame_coeffs : np.ndarray
            MDCT coefficients for one frame (1D array)
        bit : int
            Bit to embed (0 or 1)
        coef_i : int
            Index of coefficient to modify
        alpha : float
            Quantization step size
        
        Returns:
        --------
        np.ndarray
            Modified coefficients
        """
        coeffs = frame_coeffs.copy()
        
        # Preserve sign
        sign = np.sign(coeffs[coef_i]) if coeffs[coef_i] != 0 else 1
        magnitude = np.abs(coeffs[coef_i])
        
        # QIM quantization
        quantized_mag = alpha * (2 * np.round(magnitude / (2*alpha)) + (1 if bit == 1 else 0))
        coeffs[coef_i] = sign * quantized_mag
        
        return coeffs
    
    def embed_qim_bit(self, y, ind, wmbit):
        """
        Calculate dct, make precalculations, call _embed_qim_bit_helper on designated frame, perform idct. 
        
        Parameters:
        -----------
        y : list
            audio in time domain
        ind : int
            Index of frame to process
        wmbit : int
            Bit to embed (0 or 1)
        
        Returns:
        --------
        np.ndarray
            Modified coefficients
        """
        res = self.windowed_dct(y)
        num_windows, res_N = res.shape

        bins0, bins1 = self.frequency_bands(res_N)
        coefs_per_frame = bins1 - bins0

        frame_i = ind
        coef_i = self.get_coef_i(bins0, ind, coefs_per_frame)
            
        # Use the per-frame function
        res[frame_i] = self._embed_qim_bit_helper(res[frame_i], wmbit, coef_i, alpha=qim_parameters.ALPHA)

        ret = self.windowed_idct(res, y)
        return ret
    

    def embed_qim_full(self, x, alpha, bpf=1):
        """
        qim MDCT on full audio.
        """
        # mdct
        res = self.windowed_dct(x)
        num_windows, res_N = res.shape

        # frequency bands
        bins0, bins1 = self.frequency_bands(res_N)
        coefs_per_frame = bins1 - bins0
            
        # frames needed for full watermark
        frames = self.N // bpf
        for i in range(num_windows):
            b = self.watermark[i%len(self.watermark)]
            
            frame_i = i
            coef_i = self.get_coef_i(bins0, i, coefs_per_frame)
            
            # Use the per-frame function
            res[frame_i] = self._embed_qim_bit_helper(res[frame_i], b, coef_i, alpha)
        
        x = self.windowed_idct(res, x)
        return x 

    # def embed_qim_full(self, x, alpha, bpf=1):
    #     """
    #     qim MDCT on full audio.
    #     """
    #     # mdct
    #     res = self.windowed_dct(x)
    #     num_windows, res_N = res.shape

    #     # frequency bands
    #     bins0, bins1 = self.frequency_bands(res_N)
    #     coefs_per_frame = bins1 - bins0
        
    #     # frames needed for full watermark
    #     frames = self.N // bpf
        
    #     if frames > num_windows:
    #         raise ValueError(f"Audio's too short for given watermark. Adjust either or both. {frames} needed {len(res)} given.")
        
    #     for i in range(num_windows):
    #         b = self.watermark[i%len(self.watermark)]
            
    #         frame_i = i
    #         coef_i = self.get_coef_i(bins0, i, coefs_per_frame)
            
    #         # Use the per-frame function
    #         res[frame_i] = self._embed_qim_bit_helper(res[frame_i], b, coef_i, alpha)
        
    #     rev = self.windowed_idct(res, x)
    #     return rev 
    
    def _extract_qim_bit_helper(self, frame_coeffs, coef_i, alpha):
        """
        Private function to extract bit given pre-processed frames
        """
        coefficient = frame_coeffs[coef_i]
            
        # If the coefficient is closer to odd multiple of alpha, bit is 1
        # If the coefficient is closer to even multiple of alpha, bit is 0
        magnitude = np.abs(coefficient)
        quantized_value = int(np.round(magnitude / alpha))
        extracted_bit = quantized_value & 1
            
        return extracted_bit
    
    def extract_qim_bit(self, y, ind):
        mdct_coeffs = self.windowed_dct(y)
        num_windows, res_N = mdct_coeffs.shape
        
        bins0, bins1 = self.frequency_bands(res_N)

        coefs_per_frame = bins1 - bins0

        frame_i = ind
            
        coef_i = self.get_coef_i(bins0, ind, coefs_per_frame)
        extracted_bit = self._extract_qim_bit_helper(mdct_coeffs[frame_i], coef_i, alpha=qim_parameters.ALPHA)
            
        return extracted_bit
    
    def extract_watermark(self, y, bpf=1, alpha=.5):
        """
        """
        mdct_coeffs = self.windowed_dct(y)
        num_windows, res_N = mdct_coeffs.shape
        
        bins0, bins1 = self.frequency_bands(res_N)

        coefs_per_frame = bins1 - bins0
        
        extracted_bits = np.zeros(num_windows, dtype=int)
        for i in range(num_windows):
            frame_i = i
            
            coef_i = self.get_coef_i(bins0, i, coefs_per_frame)
            extracted_bit = self._extract_qim_bit_helper(mdct_coeffs[frame_i], coef_i, alpha=qim_parameters.ALPHA)
            
            extracted_bits[i] = extracted_bit
        
        return [str(exb) for exb in extracted_bits]

class Embeds:
    def __init__(self, watermark_data):
        self.time_w = Echo() 
        self.transform_w = WatermarkDCT(np.hanning, watermark_data) # need to redefine
        self.watermark_data = watermark_data

    def embed_combo_watermark(self, designations, audio, sr):
        watermark_ind = 0
        for i in range(len(designations) - 2):
            if i*hop_length >= len(audio): # just in case
                print("broken")
                print(i*hop_length)
                break

            start_ind = i*hop_length
            end_ind = start_ind+hop_length if start_ind+hop_length < len(audio) else 0
            frame = audio[start_ind:end_ind]

            wmbit = self.watermark_data[watermark_ind%len(self.watermark_data)]

            if designations[i] == domain_choice.TIME or designations[i] == domain_choice.HYBRID:
                audio[start_ind:end_ind] = self.time_w.embed_watermark_bit(frame=frame, sr=sr, wmbit=wmbit)
                

            if designations[i] == domain_choice.TRANSFORM or designations[i] == domain_choice.HYBRID:
                audio = self.transform_w.embed_qim_bit(y=audio, ind=i, wmbit=wmbit)

            watermark_ind += 1
        return audio

    def extract_combo_watermark(self, designations, audio, sr):
        watermark_ind = 0
        full_watermark = 0
        conflict = []
        for i in range(len(designations) - 2):
            if i*hop_length >= len(audio): # just in case
                break

            start_ind = i*hop_length
            end_ind = start_ind+hop_length if start_ind+hop_length < len(audio) else 0
            frame = audio[start_ind:end_ind]

            b1 = None
            b2 = None
            if designations[i] == domain_choice.TIME or designations[i] == domain_choice.HYBRID:
                b1 = self.time_w.extract_watermark_bit(frame=frame, sr=sr)
                
            if designations[i] == domain_choice.TRANSFORM or designations[i] == domain_choice.HYBRID:
                b2 = self.transform_w.extract_qim_bit(y=audio, ind=i)

            
            if b1 == None:
                full_watermark = (full_watermark << 1) + int(b2)
            elif b2 == None:
                full_watermark = (full_watermark << 1) + int(b1)
            else:
                if b1 != b2:
                    conflict.append(i)
                full_watermark = (full_watermark << 1) + int(b2)

            watermark_ind += 1
        return full_watermark



