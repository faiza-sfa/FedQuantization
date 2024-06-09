import math
from src.compressors.compressor import Compressor
import struct

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.eval("using Random; Random.seed!(0)")
Main.include("src/compressors/qsgd.jl")

from bitarray import bitarray
from bitarray.util import int2ba, ba2int
import numpy as np

class QSGD(Compressor):
    def __init__(self, s: int, zero_rle: bool, type=None):
        self.type = type
        if self.type == "LFL":
            self.s = 2
        else:
            self.s = s
        self.zero_rle = zero_rle

    # format:    | length | norm | s(1) | sign(1) | s(2) | ... | sign(n) | 
    # (no 0-rle) | 32     | 32   | ?    | 1       | ?    | ... | 1       |
    # format:    | length | norm | n_zeros | s(1) | sign(1) | n_zeros | s(2) | ... | sign(n) |
    # (0-rle)    | 32     | 32   | ?       | ?    | 1       | ?       | ?    | ... | 1       |
    def encode(self, array: np.ndarray, seed=None, cid=":-)", client_name=":-))") -> bytes:
        assert len(array.shape) == 1
        assert array.dtype == np.float32
        result = Main.encode_qsgd(array, self.s, self.type, seed, cid, client_name)
        return bytes(result)

    def decode(self, array: bytes, use_lo_quant=False) -> np.ndarray:
        result = Main.decode_qsgd(array, self.s, self.type, use_lo_quant)
        return result