from utils.evaluation.robustness_evaluation import RobustnessEvaluation

import unittest
import numpy as np
import librosa

class TestRobustnessEvaluation(unittest.TestCase):

    def setUp(self):
        self.metrics = RobustnessEvaluation()
        self.audio, self.sr = librosa.load("data/hw2_audio.wav")

    # --- detection_rate ---
    def test_detection_rate_basic(self):
        self.assertAlmostEqual(self.metrics.detection_rate(tp=50, fn=50), 0.5)

    def test_detection_rate_zero_fn(self):
        self.assertAlmostEqual(self.metrics.detection_rate(tp=10, fn=0), 1.0)

    def test_detection_rate_zero_tp(self):
        self.assertAlmostEqual(self.metrics.detection_rate(tp=0, fn=10), 0.0)

    def test_detection_rate_divide_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            self.metrics.detection_rate(tp=0, fn=0)

    # --- precision ---
    def test_precision_basic(self):
        self.assertAlmostEqual(self.metrics.precision(tp=50, fp=50), 0.5)

    def test_precision_no_fp(self):
        self.assertAlmostEqual(self.metrics.precision(tp=10, fp=0), 1.0)

    def test_precision_no_tp(self):
        self.assertAlmostEqual(self.metrics.precision(tp=0, fp=10), 0.0)

    def test_precision_avoid_zero_division(self):
        self.assertAlmostEqual(self.metrics.precision(tp=0, fp=0), 0.0)

    # --- normalized_correlation ---
    def test_normalized_correlation_perfect(self):
        real = "111000"
        extracted = "111000"
        corr = self.metrics.normalized_correlation(real=real, extracted=extracted)
        self.assertAlmostEqual(corr, 1.0)

    def test_normalized_correlation_inverse(self):
        real = "111000"
        extracted = "000111"
        corr = self.metrics.normalized_correlation(real=real, extracted=extracted)
        self.assertAlmostEqual(corr, -1.0)

    def test_normalized_correlation_random(self):
        real = "101010"
        extracted = "100110"
        corr = self.metrics.normalized_correlation(real=real, extracted=extracted)
        self.assertTrue(-1 <= corr <= 1)

    # def test_normalized_correlation_uniform(self): # failing TODO
    #     real = "111000"
    #     extracted = "000000"
    #     corr = self.metrics.normalized_correlation(real=real, extracted=extracted)
    #     self.assertAlmostEqual(corr, .5)

    # --- bit_error_rate ---
    def test_bit_error_rate_basic(self):
        real = "101010"
        diff = 2
        self.assertAlmostEqual(self.metrics.bit_error_rate(real=real, diff=diff), 2 / 6)

    def test_bit_error_rate_zero_errors(self):
        real = "1111"
        diff = 0
        self.assertAlmostEqual(self.metrics.bit_error_rate(real=real, diff=diff), 0.0)

    def test_bit_error_rate_all_errors(self):
        real = "1111"
        diff = 4
        self.assertAlmostEqual(self.metrics.bit_error_rate(real=real, diff=diff), 1.0)

    
    # --- reconstruct watermark from redundant embeddings ---
    def test_recover_extracted_bad(self):
        extracted = "11000010"
        real = "10110" 
        self.assertEqual(self.metrics.recover_extracted(extracted=extracted, flagbits=real), "11000")

    def test_recover_extracted_full(self):
        extracted = "110000011"
        real = "101"
        self.assertEqual(self.metrics.recover_extracted(extracted=extracted, flagbits=real), "101")
    
    def test_recover_extracted_partial(self):
        extracted = "11000"
        real = "101"
        self.assertEqual(self.metrics.recover_extracted(extracted=extracted, flagbits=real), "100")

    # --- transformation "attacks" ---
    def test_resample_shape(self):
        out = self.metrics.resample(audio=self.audio, sr=self.sr)
        self.assertEqual(out.shape, self.audio.shape)

    def test_requantize_values(self):
        out = self.metrics.requantize(audio=self.audio, sr=self.sr)
        self.assertTrue(np.all(out <= 1) and np.all(out >= -1))

    def test_noise_increases_variance(self):
        out = self.metrics.noise(audio=self.audio, sr=self.sr)
        self.assertEqual(len(out), len(self.audio))
        self.assertTrue(np.var(out) > 0)

    def test_lowpass_preserves_shape(self):
        out = self.metrics.lowpass(audio=self.audio, sr=self.sr)
        self.assertEqual(out.shape, self.audio.shape)

    def test_highpass_preserves_shape(self):
        out = self.metrics.highpass(audio=self.audio, sr=self.sr)
        self.assertEqual(out.shape, self.audio.shape)

    def test_amplitude_scaling(self):
        out = self.metrics.amplitude(audio=self.audio, sr=self.sr)
        self.assertTrue(np.allclose(out, self.audio * self.metrics.amplitude_val))


