from aqua import AQUA
import numpy as np
from typing import Dict, Tuple

from globals import ODG_THRESHOLD

class PerceptualEvaluation:
    def __init__(self):
        self.aqua = AQUA()
        
    def evaluate_watermark(
        self, 
        original_audio: np.ndarray, 
        watermarked_audio: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate watermark perceptibility by comparing original and watermarked audio.
        
        Args:
            original_audio: Original audio signal (numpy array)
            watermarked_audio: Watermarked audio signal (numpy array)
            
        Returns:
            Dictionary containing:
                - 'odg': Objective Difference Grade (-4 to 0, where 0 is imperceptible)
                - 'di': Distortion Index (0 to 4, where 0 is imperceptible)
                - 'perceptible': Boolean indicating if watermark is perceptible
        """
        # normalize length
        min_len = min(len(original_audio), len(watermarked_audio))
        original_audio = original_audio[:min_len]
        watermarked_audio = watermarked_audio[:min_len]
        
        # odg -4 (very perceptible) to 0 (imperceptible)
        odg, di = self.aqua.evaluate(original_audio, watermarked_audio)
        perceptible = odg < ODG_THRESHOLD
        
        return {
            'Objective Difference Grade': odg,
            'Distortion Index': di,
            'Perceptible': perceptible
        }