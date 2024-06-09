import math
from src.compressors.compressor import Compressor
import struct

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.eval("using Random; Random.seed!(0)")
Main.include("src/compressors/uveqfed.jl")

from bitarray import bitarray
from bitarray.util import int2ba, ba2int
import numpy as np
import torchac
import torch

MAX_LEN = 300_000

class UVeQFed(Compressor):
    def __init__(self, s_fDesRate: int):
        self.s_fDesRate = s_fDesRate

    def encode(self, array: np.ndarray, seed=None, cid=":-)", client_name=":-))") -> bytes:
        assert len(array.shape) == 1
        assert array.dtype == np.float32
        databytes, idxs, cdf = Main.encode_uveqfed(array, self.s_fDesRate, seed)
        idxs = idxs.squeeze()
        num_compressed_bytes = 0
        for i in range(0, len(idxs), MAX_LEN):
            enc_len = min(MAX_LEN, len(idxs)-i)
            dec_bytes = torchac.encode_float_cdf(
                torch.tensor(cdf, dtype=torch.float32).repeat(enc_len, 1),
                torch.tensor(idxs[i:i+enc_len], dtype=torch.int16), check_input_bounds=True)
            num_compressed_bytes += len(dec_bytes)
        num_compressed_bytes += len(cdf)*4
        return struct.pack("i", num_compressed_bytes) + bytes(databytes)

    def decode(self, array: bytes, use_lo_quant=False, seed=None, round=None) -> np.ndarray:
        num_compressed_bytes = struct.unpack("i", array[0:4])[0]
        databytes = array[4:]
        result = Main.decode_uveqfed(databytes, int(seed), self.s_fDesRate)
        return result, num_compressed_bytes