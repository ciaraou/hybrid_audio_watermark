import numpy as np

from globals import echo_parameters, hop_length

class Echo():
    
    def embed_watermark_bit(self, *, frame: np.ndarray, sr: int, wmbit: int) -> np.ndarray:
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
    
    def full_embed_watermark(self, *, y: np.ndarray, sr: int, wmbits: list[int]) -> np.ndarray:
        frames = len(y)//hop_length

        for i in range(frames):
            y[i*hop_length:(i+1)*hop_length] = self.embed_watermark_bit(frame=y[i*hop_length:(i+1)*hop_length], sr=sr, wmbit=wmbits[i%len(wmbits)])
    
        return y
    
    def extract_watermark_bit(self, *, frame: np.ndarray, sr: int) -> int:
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
    
    def full_extract_watermark(self, *, y: np.ndarray, sr: int) -> list[str]:
        frames = len(y)//hop_length

        res = []
        for i in range(frames):
            b = self.extract_watermark_bit(frame=y[i*hop_length:(i+1)*hop_length], sr=sr)
            res.append(str(b))
        return res
