import math
import struct

from bitarray import bitarray
from src.compressors.qsgd import QSGD
from src.compressors.fp8 import FP32_FORMAT, FP8_FORMAT, get_emax, get_emin, FP8
from src.compressors.no_compression import NoCompression
import unittest
import numpy as np
import matplotlib.pyplot as plt

from src.compressors.compressor import Compressor

class CompressorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.s = 100
        self.quantizers = [
            NoCompression(),
            QSGD(s=self.s, zero_rle=False),
            QSGD(s=self.s, zero_rle=True),
            QSGD(s=None, zero_rle=True, type="LFL"),
            FP8(),
        ]

    def test_encode_decode_approx_correct(self) -> None:
        for quantizer in self.quantizers:
            array = np.random.randn(1000).astype(np.float32)
            # special_num covers an edge case for the fp8: when the first
            # len(significand(FP8)) bits of the significand are set to 1, then we need to increment
            # the exponent when rounding up. (Normally, incrementing the significand suffices.)
            special_num = struct.unpack('>f', bitarray('0')+bitarray('01111111')+bitarray('1'*23))
            array = np.append(array, np.float32(special_num))
            encoded_array = quantizer.encode(array)
            decoded_array = quantizer.decode(encoded_array)
            diff = array - decoded_array
            if isinstance(quantizer, NoCompression):
                discretization_error = 0
            elif isinstance(quantizer, FP8):
                # The discretization error must be at most 1 FP8 ULP. np.spacing(np.abs(array))
                # gives 1 FP32 ULP. Multiply by 2^(23-2) to get 1 FP8 ULP.
                discretization_error = np.spacing(np.abs(array)) * 2**21
                # The above gives the correct ULP only if the encoded FP8 is not subnormal.
                # If it is subnormal, then the ULP will be too small by factor 2^k, where
                # k is the position of the first 1 in the significand of the FP8 digit.
                # To set the correct ULP, make all ULPs at least big as the FP8-subnormal
                # ULP.
                discretization_error = np.maximum(discretization_error, np.float32(2**(get_emin(FP8_FORMAT)-2)))
            elif isinstance(quantizer, QSGD):
                # Ideally, the discretization error will always be < norm/s. However, due to
                # floating-point errors that can result from the calculation of norm, and the
                # division of numbers, by norm the error could end up being slightly higher. 
                discretization_error = np.linalg.norm(array)/quantizer.s
            else:
                raise Exception("Unknown quantizer class.")
            assert all(abs(diff) <= discretization_error)

    def test_encode_decode_unbiased(self) -> None:
        # Test that the quantizer is unbiased by repeatedly encoding and
        # decoding the same value.  If the quantizer is unbiased, then the
        # average of the repetitions should converge to the original values.
        for c_idx, quantizer in enumerate(self.quantizers):
            array = np.random.randn(2).astype(np.float32)
            avg_decoded_array = np.zeros(2)
            reps = 10_000
            avgs = []
            for idx in range(reps):
                encoded_array = quantizer.encode(array)
                decoded_array = quantizer.decode(encoded_array)
                avg_decoded_array += decoded_array * (1/reps)
                avgs.append((avg_decoded_array * (reps/(idx+1)))[0])
            if c_idx == 1:
                plt.plot(range(reps), avgs)
                plt.hlines(array[0], xmin=0, xmax=reps, color='red')
                plt.savefig("plot.png")
            # Ideally, the discretization error will always be < norm/s.
            # However, due to floating-point errors that can result from the
            # calculation of norm, and the division of numbers, by norm the
            # error could end up being slightly higher.
            diff = array - avg_decoded_array
            if isinstance(quantizer, NoCompression):
                # Some error will occur in the summing and averaging of results.
                discretization_error = 1e-06
            elif isinstance(quantizer, FP8):
                # The discretization error must be at most 1 FP8 ULP. np.spacing(array) gives 1 FP32
                # ULP. Multiply by 2^(23-2) to get 1 FP8 ULP.
                discretization_error = np.spacing(np.abs(array)) * 2**21
            elif isinstance(quantizer, QSGD):
                # Ideally, the discretization error will always be < norm/s. However, due to
                # floating-point errors that can result from the calculation of norm, and the
                # division of numbers, by norm the error could end up being slightly higher. 
                discretization_error = np.linalg.norm(array)/quantizer.s
            else:
                raise Exception("Unknown quantizer class.")
            assert all(abs(diff) <= discretization_error/10)

    def test_zero_rle(self) -> None:
        quantizer = QSGD(s=self.s,zero_rle=True)
        array = np.array((([0]*1000)+[1])*10).astype(np.float32)
        encoded_array = quantizer.encode(array)
        decoded_array = quantizer.decode(encoded_array)
        diff = array - decoded_array
        # Ideally, the discretization error will always be < norm/s.
        # However, due to floating-point errors that can result from the
        # calculation of norm, and the division of numbers, by norm the
        # error could end up being slightly higher. 
        discretization_error = np.linalg.norm(array)/self.s
        assert all(abs(diff) <= discretization_error)
        assert len(encoded_array) == 48

    def test_8fp_values_are_perfectly_encoded_and_decoded(self) -> None:
        for quantizer in {FP8()}:
            for sign in [1, -1]:
                for exponent in range(get_emin(FP8_FORMAT), get_emax(FP8_FORMAT)):
                    for significand in [1.0, 1.25, 1.5, 1.75]:
                        num = sign * 2**exponent * significand
                        enc = quantizer.encode(np.array([num]).astype(np.float32))
                        assert len(enc) == 1
                        assert quantizer.decode(enc) == np.array([num])

    def test_8fp_special_cases_search_implementation(self) -> None:
        def assert_exception_on_encoding_num(num):
            self.assertRaises(Exception, quantizer.encode, np.array([num]).astype(np.float32))
        quantizer = FP8()
        # Case 0:
        assert quantizer.decode(quantizer.encode(np.array([0]).astype(np.float32))) \
            == np.array([0]).astype(np.float32)
        # Case smallest FP32 above the max FP8:
        num = np.nextafter(np.float32(2**get_emax(FP8_FORMAT) * 1.75), np.float32(np.Inf))
        assert quantizer.decode(quantizer.encode(np.array([num]).astype(np.float32))) \
            == 2**get_emax(FP8_FORMAT) * 1.75
        # Case greatest FP32 below the min FP8 above 0:
        num = np.nextafter(np.float32(2**get_emin(FP8_FORMAT) * 0.25), np.float32(-np.Inf))
        assert quantizer.decode(quantizer.encode(np.array([num]).astype(np.float32))) \
            == 2**get_emin(FP8_FORMAT) * 0.25
        # Case Inf:
        assert_exception_on_encoding_num(np.Inf)
        # Case -Inf:
        assert_exception_on_encoding_num(-np.Inf)
        # Case NaN:
        assert_exception_on_encoding_num(np.NaN)
        # Case FP32-subnormal:
        num = np.float32(2**(get_emin(FP32_FORMAT)-1))
        assert quantizer.decode(quantizer.encode(np.array([num]).astype(np.float32))) == 0

if __name__ == "__main__":
    unittest.main(verbosity=2)
