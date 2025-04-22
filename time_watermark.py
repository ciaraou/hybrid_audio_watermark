import numpy as np
import util as watermark_utils

class LSB():
    def __init__(self, key, watermark):
        """
        
        Parameters
        ----------
        key : array_like
            list of tuples. the key acts as indexing for watermark bits. 
            [(sample_index_0, bit_index_0), (sample_index_1, bit_index_1)... (sample_index_n, bit_index_n)] 
            Right now all bit_index should default to 0 for rightmost, and all sample_index are chosen randomly. 
            TODO: Both can be calculated for optimization later. 

        watermark : array_like
            1d array of watermark data
        """

        self.key = key
        self.watermark = watermark

    # def float_to_byte(self, value):
    #     value_bytes = value
    #     if not self.bigendian:  # if it's little endian, swap byte order
    #         value_bytes = value_bytes.byteswap()
    #     return value_bytes.tobytes()
    
    def embed_watermark(self, x):
        """
        
        Parameters
        ----------
        x : array_like
            1d host audio array
        """
        y = np.copy(x)
        for wmbit, ind in zip(self.watermark, self.key):
            # print(ind)
            sample = x[ind[0] % len(x)]
            # print(sample)
            # print(watermark_utils.float64_to_binary(sample))
            sample = watermark_utils.bit_change(sample, wmbit, ind[1])# if wmbit else self.bit0(sample, ind[1])
            # print(watermark_utils.float64_to_binary(sample))
            
            y[ind[0] % len(x)] = sample
            # print(watermark_utils.float64_to_binary(y[ind[0]]))
            # print("")

        return y
    
    def extract_watermark(self, y):
        x = np.copy(y)
        extracted_watermark = np.zeros(len(self.watermark), dtype=int)
        for i in range(len(self.key)):
            ind = self.key[i]
            sample = y[ind[0] % len(y)]
            # print(watermark_utils.float64_to_binary(sample))
            extracted_watermark[i] = watermark_utils.float64_to_binary(sample)[-((ind[1]%watermark_utils.FLOAT64_BITSIZE)+1)] # index from right
            x[ind[0] % len(y)] = sample

        return x, ''.join(str(bit) for bit in extracted_watermark)