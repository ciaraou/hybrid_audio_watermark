import numpy as np

from utils.watermark.dct_transform import DCT
from utils.watermark.window import Window

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

    # currently Quantization Index Modulation
    def apply_watermark(self, x, bpf=1, alpha=.5):
        """
        TODO try different embed methods - QIM basically takes two different quantizers, one for each bit, and selectively quantizes w it

        Parameters
        ----------
        x : array_like
            audio signal
        bpf : int
            bits per frame default 1 
        alpha : float
            quantization step size
        
        Returns
        -------
        array_like
            watermarked signal

        Raises
        ------
        ValueError
        """
        res = self.windowed_dct(x)
        num_windows, res_N = res.shape

        bins0, bins1 = self.frequency_bands(res_N)

        coefs_per_frame = bins1 - bins0 
        bits_per_frame = bpf
        frames = self.N // bits_per_frame

        if frames > len(res):
            raise ValueError(f"Audio's too short for given watermark. Adjust either or both. {frames} needed {len(res)} given.")
        
        for i in range(len(self.watermark)):
            b = self.watermark[i]
            if i >= frames: # end of watermark reached
                break

            frame_i = i 

            # coefficient selection
            coef_i = self.get_coef_i(bins0, i, coefs_per_frame)

            res[frame_i, coef_i] = alpha * (2 * np.round(res[frame_i, coef_i] / (2*alpha)) + (1 if b==1 else 0)) # + 1 if bit is 1, + 0 if bit is 0.

        rev = self.windowed_idct(res, x)
        return rev
    
    def extract_watermark(self, y, watermark_length, bpf=1, alpha=.5):
        """
        TODO try different embed methods - QIM basically takes two different quantizers, one for each bit, and selectively quantizes w it

        Parameters
        ----------
        y : array_like
            watermarked audio signal
        bpf : int
            bits per frame default 1 
        
        Returns
        -------
        tuple
            de-watermarked host signal and watermark

        Raises
        ------
        ValueError
        """
        mdct_coeffs = self.windowed_dct(y)
        num_windows, res_N = mdct_coeffs.shape
        
        bins0, bins1 = self.frequency_bands(res_N)

        coefs_per_frame = bins1 - bins0
        bits_per_frame = bpf
        frames = self.N // bits_per_frame
        
        if frames > num_windows:
            print(f"WARNING: Audio too short for expected watermark. Only extracted {min(num_windows, watermark_length)} bits.")
            max_bits = min(num_windows, watermark_length)
        else:
            max_bits = min(frames, watermark_length)
        
        extracted_bits = np.zeros(watermark_length, dtype=int)
        for i in range(max_bits):
            frame_i = i
            
            coef_i = self.get_coef_i(bins0, i, coefs_per_frame)
            coefficient = mdct_coeffs[frame_i, coef_i]
            
            # If the coefficient is closer to odd multiple of alpha, bit is 1
            # If the coefficient is closer to even multiple of alpha, bit is 0
            quantized_value = np.round(coefficient / alpha)
            extracted_bit = quantized_value % 2  # 1 if odd, 0 if even
            
            extracted_bits[i] = extracted_bit
        
        return [str(exb) for exb in extracted_bits]