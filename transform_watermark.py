import utils.watermark.util as watermark_utils
from utils.watermark.dct_transform import DCT
from deprecated.window import Window
from globals import qim_parameters

import numpy as np

class WatermarkDCT(DCT):
    """
    (M)DCT transformation domain watermark 
    """

    def __init__(self, *, watermark: list[str]) -> None: # maybe make watermark optional, random gen? TODO add win_type option if needed
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
        super().__init__()

    def frequency_bands(self, res_N_fb: int) -> tuple[int, int]:
        return res_N_fb // 4, res_N_fb // 2
    
    def get_coef_i(self, bins0_gci: int, i_gci: int, coefs_per_frame_gci: int) -> int:
        return bins0_gci + (i_gci % coefs_per_frame_gci)


    def _embed_qim_bit_helper(self, frame_coeffs: np.ndarray, bit: int, coef_i: int, alpha: float) -> np.ndarray:
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
    
    def embed_qim_bit(self, y: np.ndarray, ind: int, wmbit: int) -> np.ndarray:
        """
        Calculate dct, make precalculations, call _embed_qim_bit_helper on designated frame, perform idct. 
        
        Parameters:
        -----------
        y : np.ndarray
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
        res = self.windowed_dct(x=y)
        num_windows, res_N = res.shape

        bins0, bins1 = self.frequency_bands(res_N_fb=res_N)
        coefs_per_frame = bins1 - bins0

        frame_i = ind
        coef_i = self.get_coef_i(bins0_gci=bins0, i_gci=ind, coefs_per_frame_gci=coefs_per_frame)
            
        # Use the per-frame function
        res[frame_i] = self._embed_qim_bit_helper(frame_coeffs=res[frame_i], bit=wmbit, coef_i=coef_i, alpha=qim_parameters.ALPHA)

        ret = self.windowed_idct(frames=res, x=y)
        return ret
    

    def embed_qim_full(self, x: np.ndarray, alpha: float = qim_parameters.ALPHA, bpf: int = 1) -> np.ndarray:
        """
        qim MDCT on full audio.
        """
        # mdct
        res = self.windowed_dct(x=x)
        num_windows, res_N = res.shape

        # frequency bands
        bins0, bins1 = self.frequency_bands(res_N_fb=res_N)
        coefs_per_frame = bins1 - bins0
            
        # frames needed for full watermark
        frames = self.N // bpf
        for i in range(num_windows):
            b = self.watermark[i%len(self.watermark)]
            
            frame_i = i
            coef_i = self.get_coef_i(bins0_gci=bins0, i_gci=i, coefs_per_frame_gci=coefs_per_frame)
            
            # Use the per-frame function
            res[frame_i] = self._embed_qim_bit_helper(frame_coeffs=res[frame_i], bit=b, coef_i=coef_i, alpha=alpha)
        
        x = self.windowed_idct(frames=res, x=x)
        return x 
    
    def _extract_qim_bit_helper(self, frame_coeffs: list, coef_i: int, alpha: float) -> int:
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
    
    def extract_qim_bit(self, y: np.ndarray, ind: int) -> int:
        dct_coeffs = self.windowed_dct(x=y)
        num_windows, res_N = dct_coeffs.shape
        
        bins0, bins1 = self.frequency_bands(res_N_fb=res_N)

        coefs_per_frame = bins1 - bins0

        frame_i = ind
            
        coef_i = self.get_coef_i(bins0_gci=bins0, i_gci=ind, coefs_per_frame_gci=coefs_per_frame)
        extracted_bit = self._extract_qim_bit_helper(frame_coeffs=dct_coeffs[frame_i], coef_i=coef_i, alpha=qim_parameters.ALPHA)
            
        return extracted_bit
    
    def extract_watermark(self, y: np.ndarray, bpf: int = 1, alpha: float = qim_parameters.ALPHA) -> list[str]:
        """
        """
        dct_coeffs = self.windowed_dct(x=y)
        num_windows, res_N = dct_coeffs.shape
        
        bins0, bins1 = self.frequency_bands(res_N_fb=res_N)

        coefs_per_frame = bins1 - bins0
        
        extracted_bits = np.zeros(num_windows, dtype=int)
        for i in range(num_windows):
            frame_i = i
            
            coef_i = self.get_coef_i(bins0_gci=bins0, i_gci=i, coefs_per_frame_gci=coefs_per_frame)
            extracted_bit = self._extract_qim_bit_helper(frame_coeffs=dct_coeffs[frame_i], coef_i=coef_i, alpha=qim_parameters.ALPHA)
            
            extracted_bits[i] = extracted_bit
        
        return [str(exb) for exb in extracted_bits]
