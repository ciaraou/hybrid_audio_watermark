import librosa 
from globals import n_fft, hop_length, smoothing_window, THRESHOLD, domain_choice

# flatness
def domain_flatness(y):
    return librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]

# harmonicity 
def domain_pitch_stability(y):
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


def domain_long_term_energy(y):
    # RMS energy per frame
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]

    # Long-term energy average (moving average)
    cumsum = np.cumsum(np.insert(rms, 0, 0))
    return (cumsum[smoothing_window:] - cumsum[:-smoothing_window]) / smoothing_window

def domain_zcr(y):
    return librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]

def domain_centroid(y):
    return librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]

def determine_domain(lte, flatness, ps, zcr, centroid):
    if lte < THRESHOLD["lte"]:
        return domain_choice.NONE
    
    if flatness > THRESHOLD["flatness"] and (zcr > THRESHOLD["zcr"] or centroid > THRESHOLD["centroid"]):
        return domain_choice.TIME 
    
    if flatness < THRESHOLD["flatness"] and ps < THRESHOLD["stability"]:
        return domain_choice.TRANSFORM
    
    return domain_choice.HYBRID

def precalculate_frame_domains():
    flatness = domain_flatness(audio)
    lte = domain_long_term_energy(audio)
    ps = domain_pitch_stability(audio)
    zcr = domain_zcr(audio)
    centroid = domain_centroid(audio)

    min_len = min([len(arr) for arr in [flatness, lte, ps, zcr, centroid]])

    domain_list = []

    domain_list = {}

    for i in range(min_len):
        # domain_list.append(determine_domain(lte[i], flatness[i], ps[i]))
        s = determine_domain(lte[i], flatness[i], ps[i], zcr[i], centroid[i])
        domain_list[s] = domain_list.get(s, 0) + 1

    print(domain_list)

precalculate_frame_domains()

