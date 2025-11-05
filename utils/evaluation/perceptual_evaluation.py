import aquatk
import numpy as np
from typing import Tuple
from aquatk.metrics import PEAQ

from globals import ODG_THRESHOLD

class PerceptualEvaluation:
    def evaluate_watermark(
        self, 
        *,
        original_audio: np.ndarray, 
        watermarked_audio: np.ndarray,
        sr: int
    ) -> dict[str, float]:
        """
        Evaluate watermark perceptibility by comparing original and watermarked audio.
        
        Args:
            original_audio: Original audio signal (numpy array)
            watermarked_audio: Watermarked audio signal (numpy array)
            
        Returns:
            Dictionary containing:
                - 'odg': Objective Difference Grade (-4 to 0, where 0 is imperceptible)
                - 'di': Distortion Index (0 to 4, where 0 is imperceptible)
        """
        # normalize length
        min_len = min(len(original_audio), len(watermarked_audio))
        original_audio = original_audio[:min_len]
        watermarked_audio = watermarked_audio[:min_len]

        def normalize_rms(audio, target_rms=0.1):
            current_rms = np.sqrt(np.mean(audio**2))
            return audio * (target_rms / (current_rms + 1e-10))

        original_audio_norm = normalize_rms(original_audio)
        watermarked_audio_norm = normalize_rms(watermarked_audio)
        
        # odg -4 (very perceptible) to 0 (imperceptible)
        res = PEAQ.peaq_basic.process_audio_data(original_audio_norm, sr, watermarked_audio_norm, sr)
        perceptible = res["Objective Difference Grade"] < ODG_THRESHOLD
        # res["Perceptible"] = perceptible
        
        return res