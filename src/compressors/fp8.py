import struct
from typing import Tuple
from bitarray import bitarray
from src.compressors.compressor import Compressor
import numpy as np
print("Starting up Julia.")
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.eval("using Random; Random.seed!(0)")
Main.include("src/compressors/fp8.jl")
print("Finished starting up Julia.")

FP8_FORMAT = (1, 5, 2)
FP32_FORMAT = (1, 8, 23)

def get_emax(format):
    return (2**(format[1]-1)) - 1

def get_emin(format):
    return 1 - get_emax(format)


class FP8(Compressor):
    def __init__(self):
        self.fp8s_repr_in_fp32 = []
        self.fp8s = []
        self.s = -1
        # negative values before positive values.
        def insert(num):
            byte = struct.pack('>B', num)
            [num] = self.decode(byte)
            if not np.isnan(num):
                self.fp8s.append(byte)
                self.fp8s_repr_in_fp32.append(num)
                bits = bitarray()
                bits.frombytes(byte)
        for i in list(reversed(range(128, 253))) + list(range(0, 128)):
            insert(i)
        self.fp8s_repr_in_fp32 = np.array(self.fp8s_repr_in_fp32).astype(np.float32)

    def get_fp8_neighbors(self, f: np.float32) -> Tuple[bytes, bytes]:
        idx_high = np.searchsorted(self.fp8s_repr_in_fp32, f, side='right')
        idx_low = idx_high - 1
        if idx_high == len(self.fp8s_repr_in_fp32):
            idx_high -= 1
        return self.fp8s[idx_low], self.fp8s[idx_high], self.fp8s_repr_in_fp32[idx_low], self.fp8s_repr_in_fp32[idx_high]

    def encode(self, array: np.ndarray, seed=None, cid=":-)", client_name=":-))") -> bytes:
        assert len(array.shape) == 1
        assert array.dtype == np.float32
        result = Main.encode_fp8(array, self.fp8s_repr_in_fp32, self.fp8s, seed)
        return bytes(result)
    
    def decode(self, array: bytes, use_lo_quant=False) -> np.ndarray:
        result = Main.decode_fp8(array)
        return result
