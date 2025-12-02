import numpy as np
from scipy.fft import dct, idct
from librosa import util

from globals import n_fft, hop_length

class DCT():
    """
    (Modified) Discrete Cosine Transform (modified preferred for audio)

    slow MDCT is a slower, more detailed and customizable operation, but it runs in O(n^2) and mdct library doesn't support axis operation
    fast MDCT is fft based DCT with specific windowing, less accurate (to a small degree), runs in O(nlogn) and supports axis operation

    Purpose of making an object for this is to ensure same window, frame length, and hop length across multiple operations. 
    Right now most of the functions just call external libraries with object variables, so it's just acting as a basic wrapper.

    TODO: will likely customize them to better fit the watermark goal specifically as it gets optimized.

    Attributes:
        window: Window object
        _frame_buffer: calls librosa.util.frame 
        windowed_mdct: calculates and outputs windowed mdct transformation
    """

    def _frame_buffer(self, *, x: np.ndarray) -> np.ndarray:
        pad_amount = (hop_length - (len(x) % hop_length)) % hop_length
        x_padded = np.pad(x, (0, pad_amount))
        return util.frame(x_padded, frame_length=hop_length*2, hop_length=hop_length, axis=0)

    # def _deframe_buffer(self, frames, x):
    #     # todo make complimentary to _frame_buffer if it changes
    #     N = n_fft

    #     original_length = len(x)
    #     res = np.zeros(original_length)
    #     overlap = np.zeros(original_length) 

    #     for i in range(len(frames)):
    #         curr = i * hop_length
    #         if curr + 2*N <= original_length: # have not reached end 
    #             res[curr:curr+2*N] += frames[i]
    #             overlap[curr:curr+2*N] += np.ones(2*N)
    #         else:    # have reached end, ignore extra
    #             res[curr:] += frames[i][:original_length-curr]
    #             overlap[curr:] += np.ones(original_length - curr)
            
    #     overlap[overlap == 0] = 1
    #     res /= overlap

    #     return res
    
    def windowed_dct(self, *, x: np.ndarray) -> tuple:
        if len(x.shape) != 1:
            print("expecting 1d audio buffer.")
            return -1
        
        # Frame the audio
        frames = self._frame_buffer(x=x)
        
        # Apply window (sine window for MDCT)
        # Assuming frames shape is (num_frames, frame_length)
        window = np.sin(np.pi * (np.arange(frames.shape[1]) + 0.5) / frames.shape[1])
        windowed_frames = frames * window[np.newaxis, :]
        
        # Apply DCT (Type-IV for proper MDCT)
        # Note: scipy's dct type 4 is the MDCT
        return dct(windowed_frames, type=4, axis=1, norm='ortho')

    def windowed_idct(self, *, frames: tuple, x: np.ndarray) -> np.ndarray:
        if len(frames.shape) != 2:
            print("expecting 2d audio buffer frames")
            return -1
        
        # Apply inverse DCT (Type-IV, which is its own inverse)
        time_frames = idct(frames, type=4, axis=1, norm='ortho')
        
        # Apply window again (MDCT property: same window for analysis and synthesis)
        window = np.sin(np.pi * (np.arange(time_frames.shape[1]) + 0.5) / time_frames.shape[1])
        windowed_frames = time_frames * window[np.newaxis, :]
        
        # Overlap-add reconstruction
        return self._overlap_add(frames=windowed_frames, original_length=len(x))

    def _overlap_add(self, *, frames: tuple, original_length: int) -> np.ndarray:
        """
        Overlap-add reconstruction for MDCT frames.
        Assumes 50% overlap, global hop_length use
        """
        num_frames, frame_length = frames.shape
        if frame_length != n_fft:
            raise Exception("problem with lengths")
        
        # Allocate output buffer
        output_length = (num_frames - 1) * hop_length + frame_length
        output = np.zeros(output_length)
        
        # Overlap-add each frame
        for i, frame in enumerate(frames):
            start = i * hop_length
            output[start:start + frame_length] += frame
        
        # Trim to original length 
        return output[:original_length]
    