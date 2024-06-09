from src.compressors.compressor import Compressor

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.eval("using Random; Random.seed!(0)")
Main.include("src/compressors/gzip.jl")

import numpy as np

class GZip(Compressor):
    def __init__(self, s: int):
        self.s = s

    def encode(self, array: np.ndarray, seed=None, cid=":-)", client_name=":-))") -> bytes:
        assert len(array.shape) == 1
        assert array.dtype == np.float32
        result = Main.encode_gzip(array, self.s, seed)
        return bytes(result)

    def decode(self, array: bytes, use_lo_quant=False) -> np.ndarray:
        result = Main.decode_gzip(array, self.s)
        return result