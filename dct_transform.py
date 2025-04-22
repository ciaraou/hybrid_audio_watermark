import numpy as np
from scipy.fft import dct, idct
from librosa import util
# from mdct import mdct, imdct
# temp:
from mdct_library_functions import mdct, imdct

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
        self.hop_length = self.N # for 50% overlap

    def frame_buffer(self, x):
        # todo add option to change
        return util.frame(x, frame_length=self.N*2, hop_length=self.hop_length, axis=0, writeable=True) # check axis
    
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
        
        # frame
        frames = self.frame_buffer(x)

        # TODO window

        # transform
        # todo: look at types 
        return dct(frames, axis=1)

    def windowed_idct(self, frames, x):
        if len(frames.shape) != 2:
            print("expecting 2d audio buffer frames")
            return -1 
        
        # de-transform and de-frame
        res = idct(frames, axis=1)

        # TODO window

        return self.deframe_buffer(res, x)
    
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

