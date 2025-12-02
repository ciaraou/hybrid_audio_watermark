import librosa 
from globals import n_fft, hop_length, smoothing_window, THRESHOLD, domain_choice, echo_parameters, qim_parameters
import numpy as np

from transform_watermark import WatermarkDCT
from time_watermark import Echo

# flatness
def domain_flatness(*, y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]

# harmonicity 
def domain_pitch_stability(*, y: np.ndarray, sr: int) -> np.ndarray:
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


def domain_long_term_energy(*, y: np.ndarray, sr: int) -> np.ndarray:
    # RMS energy per frame
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]

    # Long-term energy average (moving average)
    window = np.ones(smoothing_window) / smoothing_window
    smoothed = np.convolve(rms, window, mode='same')
    return smoothed 

def domain_zcr(*, y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]

def domain_centroid(*, y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]

def determine_domain(*, lte: float, flatness: float, ps: float, zcr: float, centroid: float) -> domain_choice:
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

def precalculate_frame_domains(*, audio: np.ndarray, sr: int) -> list[domain_choice]:
    flatness = domain_flatness(y=audio, sr=sr)
    lte = domain_long_term_energy(y=audio, sr=sr)
    ps = domain_pitch_stability(y=audio, sr=sr)
    zcr = domain_zcr(y=audio, sr=sr)
    centroid = domain_centroid(y=audio, sr=sr)

    min_len = min([len(arr) for arr in [flatness, lte, ps, zcr, centroid]])

    domain_list = []

    for i in range(min_len):
        domain_list.append(determine_domain(lte=lte[i], flatness=flatness[i], ps=ps[i], zcr=zcr[i], centroid=centroid[i]))

    return domain_list

class Embeds:
    def __init__(self, *, watermark_data: list[str]) -> None:
        self.time_w = Echo() 
        self.transform_w = WatermarkDCT(watermark=watermark_data) 
        self.watermark_data = watermark_data

    def embed_combo_watermark(self, *, designations: list[domain_choice], audio: np.ndarray, sr: int) -> np.ndarray:
        audio = audio.copy()
        
        # Batch transform embeddings
        transform_frames_to_embed = []
        for i in range(len(designations) - 1):
            if i*hop_length >= len(audio):
                break
            if designations[i] == domain_choice.TRANSFORM or designations[i] == domain_choice.HYBRID:
                wmbit = int(self.watermark_data[i%len(self.watermark_data)])
                transform_frames_to_embed.append((i, wmbit))
        
        # all transform frames
        if transform_frames_to_embed:
            # dct transformation and precalculations
            dct_coeffs = self.transform_w.windowed_dct(x=audio)
            if isinstance(dct_coeffs, int) and dct_coeffs == -1:
                raise ValueError("DCT failed: expected 1D audio buffer")
            num_windows, res_N = dct_coeffs.shape
            bins0, bins1 = self.transform_w.frequency_bands(res_N_fb=res_N)
            coefs_per_frame = bins1 - bins0
            
            # modify
            for frame_i, wmbit in transform_frames_to_embed:
                if frame_i >= num_windows:
                    continue
                coef_i = self.transform_w.get_coef_i(bins0_gci=bins0, i_gci=frame_i, coefs_per_frame_gci=coefs_per_frame)
                dct_coeffs[frame_i] = self.transform_w._embed_qim_bit_helper(
                    frame_coeffs=dct_coeffs[frame_i], 
                    bit=wmbit, 
                    coef_i=coef_i, 
                    alpha=qim_parameters.ALPHA
                )
            
            # idct
            audio = self.transform_w.windowed_idct(frames=dct_coeffs, x=audio)
        
        # time embeddings
        for i in range(len(designations) - 1):
            if i*hop_length >= len(audio):
                break

            start_ind = i*hop_length
            end_ind = start_ind+hop_length if start_ind+hop_length < len(audio) else len(audio)
            frame = audio[start_ind:end_ind]

            wmbit = int(self.watermark_data[i%len(self.watermark_data)])

            if designations[i] == domain_choice.TIME or designations[i] == domain_choice.HYBRID:
                audio[start_ind:end_ind] = self.time_w.embed_watermark_bit(frame=frame, sr=sr, wmbit=wmbit)

        return audio

    def extract_combo_watermark(self, *, designations: list[domain_choice], audio: np.ndarray, sr: int) -> list[str]:
        # Batch transform extractions
        transform_frames_to_extract = []
        for i in range(len(designations) - 1):
            if i*hop_length >= len(audio):
                break
            if designations[i] == domain_choice.TRANSFORM or designations[i] == domain_choice.HYBRID:
                transform_frames_to_extract.append(i)
        
        # transform extractions
        transform_extractions = {}
        if transform_frames_to_extract:
            # dct transformation and precalculations
            dct_coeffs = self.transform_w.windowed_dct(x=audio)
            if isinstance(dct_coeffs, int) and dct_coeffs == -1:
                raise ValueError("DCT failed: expected 1D audio buffer")
            num_windows, res_N = dct_coeffs.shape
            bins0, bins1 = self.transform_w.frequency_bands(res_N_fb=res_N)
            coefs_per_frame = bins1 - bins0
            
            # extract
            for frame_i in transform_frames_to_extract:
                if frame_i >= num_windows:
                    continue
                coef_i = self.transform_w.get_coef_i(bins0_gci=bins0, i_gci=frame_i, coefs_per_frame_gci=coefs_per_frame)
                bit = self.transform_w._extract_qim_bit_helper(frame_coeffs=dct_coeffs[frame_i], coef_i=coef_i, alpha=qim_parameters.ALPHA)
                transform_extractions[frame_i] = bit
        
        # time extractions
        full_watermark = []
        for i in range(len(designations) - 1):
            if i*hop_length >= len(audio): # just in case
                break

            start_ind = i*hop_length
            end_ind = start_ind+hop_length if start_ind+hop_length < len(audio) else len(audio)
            frame = audio[start_ind:end_ind]

            # do all
            if designations[i] == domain_choice.TIME:
                bit = self.time_w.extract_watermark_bit(frame=frame, sr=sr)
                full_watermark.append(str(bit))
            elif designations[i] == domain_choice.TRANSFORM:
                full_watermark.append(str(transform_extractions[i]))
            # elif designations[i] == domain_choice.HYBRID: # keeping commented in case I add back layered frames
            #     b1 = self.time_w.extract_watermark_bit(frame=frame, sr=sr)
            #     b2 = transform_extractions.get(i, 0)
            #     full_watermark.append(str(b2))
            # else:  # NONE or unknown
            #     full_watermark.append('0')

        return full_watermark



