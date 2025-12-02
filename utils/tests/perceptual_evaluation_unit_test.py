from utils.evaluation.perceptual_evaluation import PerceptualEvaluation
from utils.watermark.util import tobits
from aquatk.metrics import PEAQ
from globals import metric_names_p
from combo import Embeds, precalculate_frame_domains
from time_watermark import Echo

import unittest
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from pedalboard import Pedalboard, Delay

# guess: echo makes it sound like more reverb which is generaly regarded as good
# alt guess: normalization from echo watermark makes it think there's less snr
# alt guess 2: pedalboard echo doesnt add.

METRIC_NAME = "Objective Difference Grade"
 
class TestPerceptualEvaluation(unittest.TestCase):

    def _lowpass_for_test(self, *, audio: np.ndarray, sr: int, lowpass_val: int, order: int = 2) -> np.ndarray:
        # same as robustness_evaluation.py
        nyquist = 0.5 * sr
        normalized_cutoff = lowpass_val / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, audio)
    
    def _simple_echo(self, *, audio, sr, delay_seconds=0.1, alpha=0.5):
        delay_samples = int(delay_seconds * sr)
        echoed = audio.copy()
        
        # Add delayed version to original
        echoed[delay_samples:] += alpha * audio[:-delay_samples]
        return echoed

    def _expecting_usual_result(self, *, audio1: np.ndarray, audio2: np.ndarray, metric: str="Objective Difference Grade") -> dict[str, float]:
        aquatk = PEAQ.peaq_basic.process_audio_data(ref_audio=audio1, ref_rate=self.sr, test_audio=audio2, test_rate=self.sr)
        assert aquatk[metric] <= 0 
        assert aquatk[metric] >= -4
        return aquatk

    def _expecting_unusual_result(self, *, audio1: np.ndarray, audio2: np.ndarray, metric: str="Objective Difference Grade") -> dict[str, float]:
        aquatk = PEAQ.peaq_basic.process_audio_data(ref_audio=audio1, ref_rate=self.sr, test_audio=audio2, test_rate=self.sr)
        assert aquatk[metric] > 0
        return aquatk

    def setUp(self):
        self.metrics = PerceptualEvaluation()
        # base audio
        self.audio, self.sr = librosa.load("data/hw2_audio.wav", sr = None)
        # lowpassed
        self.audio_lowpassed = self._lowpass_for_test(audio=self.audio, sr=self.sr, lowpass_val=6000)
        
        # pedalboard echo
        board = Pedalboard([
            Delay(delay_seconds=0.01, feedback=0, mix=.1) # 01 to 05
        ])
        self.audio_pedalboard_echo = board(self.audio, self.sr)

        # simple echo
        self.audio_simple_echo = self._simple_echo(audio=self.audio, sr=self.sr)

        # echo watermark
        wm = tobits(s="Something to test")
        tmp_echo = Echo()
        self.audio_echo_watermarked = tmp_echo.full_embed_watermark(y=self.audio, sr=self.sr, wmbits=wm)

        # lowpass + pedalboard
        self.audio_lowpass_echoed = board(self.audio_lowpassed, self.sr)

        # lowpass + watermark
        self.audio_lowpass_watermark = tmp_echo.full_embed_watermark(y=self.audio_lowpassed, sr=self.sr, wmbits=wm)

        # lowpass + simple echo
        self.audio_lowpass_simple_echo = self._simple_echo(audio=self.audio_lowpassed, sr=self.sr)

    # check
    def test_pedalboard_is_not_additive(self):
        # check pedalboard changes the volume, or normalizes to a different range
        assert np.max(np.abs(self.audio)) != np.max(np.abs(self.audio_pedalboard_echo))

    # base: audio vs something
    def test_lowpassed_audio_is_evaluated_as_worse(self):
        # audio vs lowpassed
        # expect odg to be -4 to 0. lowpass degrades audio quality, so should be in that range. 
        self._expecting_usual_result(audio1=self.audio, audio2=self.audio_lowpassed)
    
    def test_audio_vs_normalized(self):
        # audio vs audio quieter
        # echo watermark gets normalized by a small amount, since echo could take it over the -1 to 1 range. 
        # test asserts that dividing by 1.000000000001 and evaluating that audio gets a positive result
        audio_quiet = self.audio / 1.000000000001 # 1.000000000001 got .2
        self._expecting_unusual_result(audio1=self.audio, audio2=audio_quiet)

    def test_audio_vs_pedalboard_echo(self):
        # audio vs echoed
        self._expecting_usual_result(audio1=self.audio, audio2=self.audio_pedalboard_echo)

        # audio vs quieter echoed
        normalized_echo = self.audio_pedalboard_echo / 1.000000000001
        self._expecting_usual_result(audio1=self.audio, audio2=normalized_echo)

    def test_audio_vs_echo_watermark(self):
        # audio vs echo watermark
        self._expecting_unusual_result(audio1=self.audio, audio2=self.audio_echo_watermarked)

    def test_audio_vs_simple_echo(self):
        # audio vs simple echo
        self._expecting_usual_result(audio1=self.audio, audio2=self.audio_simple_echo)

        # audio vs quieter echoed
        normalized_echo = self.audio_simple_echo / 1.000000000001
        self._expecting_usual_result(audio1=self.audio, audio2=normalized_echo)

    # lowpass improvement: lowpass vs lowpass + something
    def test_lowpass_vs_lowpass_pedalboard_echo(self):
        self._expecting_usual_result(audio1=self.audio_lowpassed, audio2=self.audio_lowpass_echoed)

    def test_lowpass_vs_lowpass_echo_watermark(self):
        self._expecting_unusual_result(audio1=self.audio_lowpassed, audio2=self.audio_lowpass_watermark)

    def test_lowpass_vs_lowpass_simple_echo(self):
        self._expecting_usual_result(audio1=self.audio_lowpassed, audio2=self.audio_lowpass_simple_echo)

    def test_lowpass_vs_lowpass_normalized(self):
        normalized_echo = self.audio_lowpassed / 1.000000000001
        self._expecting_unusual_result(audio1=self.audio_lowpassed, audio2=normalized_echo)

    # lowpass improvement: audio vs lowpass + something
    def test_audio_vs_lowpass_normalized(self):
        normalized_echo = self.audio_lowpassed / 1.01
        res = self._expecting_usual_result(audio1=self.audio, audio2=normalized_echo)
        lowpass_result = PEAQ.peaq_basic.process_audio_data(ref_audio=self.audio, ref_rate=self.sr, test_audio=self.audio_lowpassed, test_rate=self.sr)
        print("lpi")
        print(res)
        print(lowpass_result)
        print("end")
        assert res[METRIC_NAME] > lowpass_result[METRIC_NAME] # some improvement

    def test_audio_vs_lowpass_echo_watermark(self):
        res = self._expecting_usual_result(audio1=self.audio, audio2=self.audio_lowpass_watermark)
        lowpass_result = PEAQ.peaq_basic.process_audio_data(ref_audio=self.audio, ref_rate=self.sr, test_audio=self.audio_lowpassed, test_rate=self.sr)
        print("echo")
        print(res)
        print(lowpass_result)
        print("end")
        assert res[METRIC_NAME] > lowpass_result[METRIC_NAME] # some improvement

    def test_audio_vs_lowpass_pedalboard_echo(self):
        res = self._expecting_usual_result(audio1=self.audio, audio2=self.audio_lowpass_echoed)
        lowpass_result = PEAQ.peaq_basic.process_audio_data(ref_audio=self.audio, ref_rate=self.sr, test_audio=self.audio_lowpassed, test_rate=self.sr)
        print("pedal")
        print(res)
        print(lowpass_result)
        assert res[METRIC_NAME] > lowpass_result[METRIC_NAME] # some improvement

    def test_audio_vs_lowpass_simple_echo(self):
        res = self._expecting_usual_result(audio1=self.audio, audio2=self.audio_lowpass_simple_echo)
        lowpass_result = PEAQ.peaq_basic.process_audio_data(ref_audio=self.audio, ref_rate=self.sr, test_audio=self.audio_lowpassed, test_rate=self.sr)
        print(res)
        print(lowpass_result)
        assert res[METRIC_NAME] > lowpass_result[METRIC_NAME] # some improvement
    
    



    # def test_aquatk_fork_edit(self):
    #     # checks the edits in the aquatk fork: does passing in librosa data act the same as passing in files
    #     ref_filename = "data/hw2_audio.wav"
    #     test_filename = "hw2_watermarked.wav"
    #     aquatk_test_flag = tobits("flag{yeah_but_also_no}")
    #     aquatk_test_w_hybrid = Embeds(aquatk_test_flag)
    #     aquatk_test_audio, aquatk_test_sr = librosa.load(ref_filename, sr=None)
    #     aquatk_test_embed_designations = precalculate_frame_domains(aquatk_test_audio, aquatk_test_sr)
    #     aquatk_test_wm_hybrid = aquatk_test_w_hybrid.embed_combo_watermark(aquatk_test_embed_designations, aquatk_test_audio, aquatk_test_sr)

    #     # comparison
    #     info = sf.info(ref_filename)
    #     sf.write(test_filename, aquatk_test_wm_hybrid, aquatk_test_sr, subtype=info.subtype)
    #     info2 = sf.info(test_filename)

    #     assert info.channels == info2.channels
    #     assert info.format == info2.format 
    #     assert info.samplerate == info2.samplerate
    #     assert info.subtype == info2.subtype

    #     # compare file vs data
    #     aquatk_test_audio, aquatk_test_sr = librosa.load(ref_filename, sr=None)
    #     aquatk_test_audio2, aquatk_test_sr2 = librosa.load(test_filename, sr=None)

    #     aquatk_test_perceptual_data = PEAQ.peaq_basic.process_audio_data(ref_audio=aquatk_test_audio, ref_rate=aquatk_test_sr, test_audio=aquatk_test_audio2, test_rate=aquatk_test_sr2)
    #     aquatk_test_perceptual_file = PEAQ.peaq_basic.process_audio_files(ref_filename=ref_filename, test_filename=test_filename)

    #     # same dict
    #     assert len(aquatk_test_perceptual_data) == len(aquatk_test_perceptual_file)
    #     assert aquatk_test_perceptual_file[metric_names_p[0]] == aquatk_test_perceptual_data[metric_names_p[0]]
    #     assert aquatk_test_perceptual_file[metric_names_p[1]] == aquatk_test_perceptual_data[metric_names_p[1]]
