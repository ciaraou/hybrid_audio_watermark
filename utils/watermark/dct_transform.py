import numpy as np
from scipy.fft import dct, idct
from librosa import util
# from mdct import mdct, imdct
# temp:
from deprecated.mdct_library_functions import mdct, imdct
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
        frame_buffer: calls librosa.util.frame 
        windowed_mdct: calculates and outputs windowed mdct transformation
    """

    def __init__(self, window):
        """
        Initializes MDCT object with window

        Parameters
        ----------
        window : Window 
            window object
        """
        self.window = window # TODO: not great organization of parameters right now - clean that up + names
        self.N = self.window.N

    def frame_buffer(self, x):
        # todo add option to change
        return util.frame(x, frame_length=hop_length*2, hop_length=hop_length, axis=0, writeable=True) # check axis
    
    def deframe_buffer(self, frames, x):
        # todo make complimentary to frame_buffer if it changes
        N = self.N

        original_length = len(x)
        res = np.zeros(original_length)
        overlap = np.zeros(original_length) # TODO: figure out a better way to do the length calculation

        for i in range(len(frames)):
            curr = i * self.hop_length
            if curr + 2*N <= original_length: # have not reached end 
                res[curr:curr+2*N] += frames[i]
                overlap[curr:curr+2*N] += np.ones(2*N)
            else:    # have reached end, ignore extra
                res[curr:] += frames[i][:original_length-curr]
                overlap[curr:] += np.ones(original_length - curr)
            
        overlap[overlap == 0] = 1
        res /= overlap

        return res
    
    def windowed_dct(self, x):
        if len(x.shape) != 1:
            print("expecting 1d audio buffer.")
            return -1
        
        # Frame the audio
        frames = self.frame_buffer(x)
        
        # Apply window (sine window for MDCT)
        # Assuming frames shape is (num_frames, frame_length)
        window = np.sin(np.pi * (np.arange(frames.shape[1]) + 0.5) / frames.shape[1])
        windowed_frames = frames * window[np.newaxis, :]
        
        # Apply DCT (Type-IV for proper MDCT)
        # Note: scipy's dct type 4 is the MDCT
        return dct(windowed_frames, type=4, axis=1, norm='ortho')

    def windowed_idct(self, frames, x):
        if len(frames.shape) != 2:
            print("expecting 2d audio buffer frames")
            return -1
        
        # Apply inverse DCT (Type-IV, which is its own inverse)
        time_frames = idct(frames, type=4, axis=1, norm='ortho')
        
        # Apply window again (MDCT property: same window for analysis and synthesis)
        window = np.sin(np.pi * (np.arange(time_frames.shape[1]) + 0.5) / time_frames.shape[1])
        windowed_frames = time_frames * window[np.newaxis, :]
        
        # Overlap-add reconstruction
        return self.overlap_add(windowed_frames, len(x))

    def overlap_add(self, frames, original_length):
        """
        Overlap-add reconstruction for MDCT frames.
        Assumes 50% overlap (hop_length = frame_length // 2)
        """
        num_frames, frame_length = frames.shape
        hop_length = frame_length // 2
        
        # Allocate output buffer
        output_length = (num_frames - 1) * hop_length + frame_length
        output = np.zeros(output_length)
        
        # Overlap-add each frame
        for i, frame in enumerate(frames):
            start = i * hop_length
            output[start:start + frame_length] += frame
        
        # Trim to original length
        return output[:original_length]
    
    # --------------------

    # not currently in use
    def windowed_mdct(self, x):
        if len(x.shape) != 1:
            print("expecting 1d audio buffer.")
            return -1
        
        # frame
        frames = self.frame_buffer(x)

        # TODO: window

        # transform
        for i in range(len(frames)):
            frame = frames[i]
            frames[i] = mdct(frame)

        return frames

    # not currently in use
    def windowed_imdct(self, frames):
        if len(frames.shape) != 2:
            print("expecting 2d audio buffer frames")
            return -1 

        # de-transform
        for i in range(len(frames)):
            frame = frames[i]
            frames[i] = imdct(frame)

        # TODO: window 

        # de-frame
        return self.deframe_buffer(frames)

