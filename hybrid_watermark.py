# Hybrid

from random import randint
from Crypto.Hash import SHA256

import util as watermark_utils
import time_watermark

SHA256_LENGTH = 256

class Hybrid():
    def __init__(self, transform):
        """
        TODO add option to give parameters instead?

        Parameters
        ----------
        transform : WatermarkDCT object

        time : LSB object
        """
        self.transform = transform
        self.key = [(randint(0, 307829), 0) for _ in range(SHA256_LENGTH)]#[(randint(0, sys.maxsize), 0) for _ in range(SHA256_LENGTH)] # saved here for testing purposes

    def embed_watermark(self, host):
        """
        
        Parameters
        ----------
        host : array_like
            host signal

        Returns
        -------
        string
            bit string real hash
        array_like
            array of float64, watermarked signal
        """
        # zerod_audio = watermark_utils.zero_lsb(host)
        output = self.transform.apply_watermark(host)
        zerod_output = watermark_utils.zero_lsb(output) # TODO: might interfere with QIM, figure out a dif way to make this blind-ish

        # ready for hashing
        ready_output = zerod_output.tobytes()

        # hash 
        audio_hash = self.sha256_audio(ready_output)

        # LSB
        lsb_watermark = time_watermark.LSB(self.key, audio_hash)

        return audio_hash, lsb_watermark, lsb_watermark.embed_watermark(zerod_output)
    
    def extract_watermark(self, host, time_watermark, transform_watermark_length):
        """
        
        Parameters
        ----------
        host : array_like
            host signal

        Returns
        -------
        exwm_lsbtest : string
            string representation of watermark bits extracted from lsb watermark (the hash)
        ready_output : bytes
            host signal with lsb watermark "removed" (lsb zeroed)
        exwmbits : string
            string representation of watermark bits extracted from dct/qim watermark (the actual watermark)
        """
        res_lsbtest, exwm_lsbtest = time_watermark.extract_watermark(host)
        zerod_output = watermark_utils.zero_lsb(res_lsbtest)
        ready_output = zerod_output.tobytes()

        extracted_watermark = self.transform.extract_watermark(zerod_output, transform_watermark_length)
        exwmbits = "".join([str(int(x)) for x in extracted_watermark])

        return exwm_lsbtest, ready_output, exwmbits
        # audio_hash_checker = self.sha256_audio(ready_output)

    def sha256_audio(self, tohash):
        """
        takes in bytes returns bit string

        Parameters
        ----------
        tohash : bytes
            something to hash

        Returns
        -------
        string
            string of bits
        """
        sha256 = SHA256.new()
        sha256.update(tohash)
        return watermark_utils.bytestring_to_bitstring(sha256.digest())