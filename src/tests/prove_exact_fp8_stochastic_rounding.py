import math
import struct

from bitarray import bitarray
from src.compressors.qsgd import QSGD
from src.compressors.fp8 import FP32_FORMAT, FP8_FORMAT, get_emax, get_emin, FP8
from src.compressors.no_compression import NoCompression
import unittest
import numpy as np
import matplotlib.pyplot as plt
from bigfloat import *

from src.compressors.compressor import Compressor

class CompressorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.s = 100
        self.quantizers = [
            NoCompression(),
            QSGD(s=self.s, zero_rle=False),
            QSGD(s=self.s, zero_rle=True),
            FP8(),
        ]

    def test_fp8_search_implementation_probability_computation_is_accurate(self) -> None:
        quantizer = FP8()
        max_fp8 = quantizer.decode(bitarray('01111011').tobytes())[0]
        print("max_fp8", max_fp8)
        fp32 = np.float32(0)
        ctr = 0
        while fp32 < max_fp8:
            _,_,low,high = quantizer.get_fp8_neighbors(fp32)
            big_fp32 = BigFloat(fp32.item(), single_precision)
            big_high = BigFloat(high.item(), single_precision)
            big_low = BigFloat(low.item(), single_precision)
            result = BigFloat((big_fp32 - big_low) / (big_high - big_low), single_precision)
            if fp32 < 0:  # Do this to avoid optimizing result away.
                print(result)
            if test_flag('Inexact'):
                print(f"Flags {get_flagstate()} set for float {fp32}.")
                assert False
            fp32 = np.nextafter(fp32, np.float32(np.Inf))
            ctr += 1
            if ctr % 1_000_0 == 0:
                print(f"ctr: {ctr}, float: {fp32}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
