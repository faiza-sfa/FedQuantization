# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 07:08:16 2024


@author: Faiza
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
import pandas as pd
import torch
from tqdm import tqdm
from load_dataset import Dataset
import os
import time
import cv2
import sys
from datetime import datetime
import math
import struct
from bitarray import bitarray
from bitarray.util import int2ba, ba2int
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
from scipy.fftpack import dct, idct
from sklearn.preprocessing import MinMaxScaler


Main.include("src/compressors/qsgd.jl")
Main.include("src/compressors/gzip.jl")
Main.include("src/compressors/fp8.jl")

np.random.seed(786)


class Compressor(ABC):
    @abstractmethod
    def encode(self, array: np.ndarray) -> bytes:
        pass
    @abstractmethod
    def decode(self, array: bytes) -> np.ndarray:
        pass


class QSGD(Compressor):
    def __init__(self, s: int, zero_rle: bool, type=None):
        np.random.seed(78563)
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


class GZip(Compressor):
    def __init__(self, s: int):
        np.random.seed(786)
        self.s = s

    def encode(self, array: np.ndarray, seed=None, cid=":-)", client_name=":-))") -> bytes:
        assert len(array.shape) == 1
        assert array.dtype == np.float32
        result = Main.encode_gzip(array, self.s, seed)
        return bytes(result)

    def decode(self, array: bytes, use_lo_quant=False) -> np.ndarray:
        result = Main.decode_gzip(array, self.s)
        return result
    


FP8_FORMAT = (1, 5, 2)
FP32_FORMAT = (1, 8, 23)

def get_emax(format):
    return (2**(format[1]-1)) - 1

def get_emin(format):
    return 1 - get_emax(format)


class FP8(Compressor):
    def __init__(self):
        np.random.seed(869)
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
    


# -----------------------------Custom Schemes------------------------------------
class Custom_linear():
    def __init__(self, quantization_type = np.uint32):
        np.random.seed(348543)
        self.quantization_type = quantization_type
    
    def encode(self,data):
        self.xp = [data.min(), data.max()]
        min = np.iinfo(self.quantization_type).min
        max = np.iinfo(self.quantization_type).max

        self.fp = [min, max]
        enc = np.interp(data, self.xp, self.fp)
        enc = enc.astype(self.quantization_type)

        return enc

    def decode(self, enc) :
        dec = np.interp(enc, self.fp, self.xp)
        return dec
    
class Custom_linear_16():
    def __init__(self, quantization_type = np.uint16):
        np.random.seed(786)
        self.quantization_type = quantization_type
    
    def encode(self,data):
        self.xp = [data.min(), data.max()]
        min = np.iinfo(self.quantization_type).min
        max = np.iinfo(self.quantization_type).max

        self.fp = [min, max]
        enc = np.interp(data, self.xp, self.fp)
        enc = enc.astype(self.quantization_type)
        return enc

    def decode(self, enc) :
        dec = np.interp(enc, self.fp, self.xp)
        return dec



# -----------------------------Custom Frequency Schemes------------------------------------
class customFreq():
    def __init__(self, precision_levels = [16, 8, 4]):
        np.random.seed(786)
        self.precision_levels = precision_levels
        

    def dct_transform(self, x):
        return dct(dct(x, axis=-1, norm='ortho'), axis=-1, norm='ortho')

    def inverse_dct_transform(self, x):
        return idct(idct(x, axis=-1, norm='ortho'), axis=-1, norm='ortho')

    def quantize(self, x, precision):
        scale = 2 ** (precision - 1) - 1
        return np.round(x * scale) / scale


    def encode(self, weights):
        # Apply DCT to transform weights to the frequency domain
        weights_f = self.dct_transform(weights)
        
        # Calculate importance as the magnitude of the frequency components
        importance = np.abs(weights_f)
        
        # Assign precision based on importance
        mean_importance = np.mean(importance)
        if mean_importance > 0.1:
            precision = self.precision_levels[0]  # 16-bit
        elif mean_importance > 0.01:
            precision = self.precision_levels[1]  # 8-bit
        else:
            precision = self.precision_levels[2]  # 4-bit
        
        # Quantize frequency components
        enc = self.quantize(weights_f, precision)
        enc = enc.astype(np.float16)
        return enc

    def decode(self, enc) :
        weights_quantized = self.inverse_dct_transform(enc)
        return weights_quantized
    
class customFreq_1_5():
    def __init__(self, precision_levels = [32, 16, 8]):
        np.random.seed(4285)
        self.precision_levels = precision_levels
        

    def dct_transform(self, x):
        return dct(dct(x, axis=-1, norm='ortho'), axis=-1, norm='ortho')

    def inverse_dct_transform(self, x):
        return idct(idct(x, axis=-1, norm='ortho'), axis=-1, norm='ortho')

    def quantize(self, x, precision):
        scale = 2 ** (precision - 1) - 1
        return np.round(x * scale) / scale


    def encode(self, weights):
        weights_f = self.dct_transform(weights)
        
        importance = np.abs(weights_f)
        mean_importance = np.mean(importance)
        if mean_importance > 0.1:
            precision = self.precision_levels[0]  # 32-bit
        elif mean_importance > 0.01:
            precision = self.precision_levels[1]  # 16-bit
        else:
            precision = self.precision_levels[2]  # 8-bit
        
        # Quantize frequency components
        enc = self.quantize(weights_f, precision)
        enc = enc.astype(np.float16)
        return enc

    def decode(self, enc) :
        weights_quantized = self.inverse_dct_transform(enc)
        return weights_quantized
    
class customFreq_singleDCT():
    # def __init__(self, precision_levels = [8, 4, 2]):
    def __init__(self, precision_levels = [32, 16, 8]):
        np.random.seed(786)
        self.precision_levels = precision_levels
        #quantization_level from 0 to 100        

    def dct_transform(self, x):
        return dct(x, axis=-1, norm='ortho')

    def inverse_dct_transform(self, x):
        return idct(x, axis=-1, norm='ortho')

    def quantize(self, x, precision):
        scale = 2 ** (precision - 1) - 1
        return np.round(x * scale) / scale


    def encode(self, weights):
        # Apply DCT to transform weights to the frequency domain
        weights_f = self.dct_transform(weights)
        
        # Calculate importance as the magnitude of the frequency components
        importance = np.abs(weights_f)
        
        # Assign precision based on importance
        mean_importance = np.mean(importance)
        if mean_importance > 0.1:
            precision = self.precision_levels[0]  # 8-bit
        elif mean_importance > 0.01:
            precision = self.precision_levels[1]  # 4-bit
        else:
            precision = self.precision_levels[2]  # 2-bit
        
        # Quantize frequency components
        enc = self.quantize(weights_f, precision)
        enc = enc.astype(np.float16)
        return enc

    def decode(self, enc) :
        weights_quantized = self.inverse_dct_transform(enc)
        return weights_quantized
    


# -----------------------------Custom Frequency + Linear Encoding------------------------------------


class Linear_quant():
    def __init__(self, quantization_type = np.uint32):
        np.random.seed(786)
        self.quantization_type = quantization_type
        
    def encode(self,data):
        self.xp = [data.min(), data.max()]
        min = np.iinfo(self.quantization_type).min
        max = np.iinfo(self.quantization_type).max

        self.fp = [min, max]
        enc = np.interp(data, self.xp, self.fp)
        enc = enc.astype(self.quantization_type)
        return enc

    def decode(self, enc) :
        dec = np.interp(enc, self.fp, self.xp)
        return dec
    
class customFreq_2():
    def __init__(self, precision_levels = [32, 16, 8]):
        np.random.seed(786)
        self.precision_levels = precision_levels
        self.linearQuant = Linear_quant()

    def dct_transform(self, x):
        return dct(dct(x, axis=-1, norm='ortho'), axis=-1, norm='ortho')

    def inverse_dct_transform(self, x):
        return idct(idct(x, axis=-1, norm='ortho'), axis=-1, norm='ortho')

    def quantize(self, x, precision):
        scale = 2 ** (precision - 1) - 1
        return np.round(x * scale) / scale


    def encode(self, weights):
        weights_f = self.dct_transform(weights)
        enc = self.linearQuant.encode(weights_f)
        return enc

    def decode(self, enc) :
        weights_quantized = self.linearQuant.decode(enc)
        weights_quantized = self.inverse_dct_transform(weights_quantized)
        return weights_quantized
    
class customFreq_2_singleDCT():
    def __init__(self, precision_levels = [32, 16, 8]):
        np.random.seed(786)
        self.precision_levels = precision_levels
        self.linearQuant = Linear_quant()
        #quantization_level from 0 to 100        

    def dct_transform(self, x):
        return dct(x, axis=-1, norm='ortho')
    def inverse_dct_transform(self, x):
        return idct(x, axis=-1, norm='ortho')
    def quantize(self, x, precision):
        scale = 2 ** (precision - 1) - 1
        return np.round(x * scale) / scale


    def encode(self, weights):
        # Apply DCT to transform weights to the frequency domain
        weights_f = self.dct_transform(weights)
        # Quantize frequency components
        enc = self.linearQuant.encode(weights_f)
        # enc = enc.astype(np.float16)
        return enc

    def decode(self, enc) :
        weights_quantized = self.linearQuant.decode(enc)
        weights_quantized = self.inverse_dct_transform(weights_quantized)
        return weights_quantized
    
def get_efficiency(cmp,data_size=(1,16),debug=False):
    data = list(np.random.rand(data_size[0],data_size[1]).flatten())
    data = np.array(data, dtype = np.float32)
    data_type = np.uint16
    enc = cmp.encode(data) #quantization_level from 0 to 100
    dec = cmp.decode(enc)
    if debug:
        print("Data: ",data)
        print("Encoded",enc)
        print("After Quantization: ", dec)
        print(f"{data_type} {100 * (sys.getsizeof(dec)-sys.getsizeof(enc))/sys.getsizeof(dec)}% smaller ")
    
    eff = (sys.getsizeof(dec)-sys.getsizeof(enc))/sys.getsizeof(dec)
    return eff

def get_time_efficiency_acc(cmp,
                            acc_loss_file="D:\MS_Thesis\Pre-defense-june\Results_for_update\scheme_ACC_LOSS_sheet.csv",
                            time_eff_file="D:\MS_Thesis\Pre-defense-june\Results_for_update\scheme_TIME_EFF_sheet.csv",
                            w_acc = 0.5, w_time = 0.3, w_eff = 0.2):

    df_acc_loss = pd.read_csv(acc_loss_file)

    acc_loss = df_acc_loss[df_acc_loss['Epoch']==9]
    acc_loss = acc_loss.drop(['Epoch'], axis=1)
    df_time_eff = pd.read_csv(time_eff_file)
    df_properties = pd.merge(acc_loss, df_time_eff, on='scheme_name')
    # print(df_properties)
    scaler = MinMaxScaler()
    df_properties_scaled = df_properties.copy()
    df_properties_scaled[['Acc', 'Time', 'Efficiency']] = scaler.fit_transform(df_properties_scaled[['Acc', 'Time', 'Efficiency']])
    weights = {
        'Acc': w_acc,
        'Time': w_time,
        'Efficiency': w_eff
    }

    # Compute the combined measure
    df_properties_scaled['Combined_Score'] = (
        weights['Acc'] * df_properties_scaled['Acc'] +
        weights['Time'] * (1 - df_properties_scaled['Time']) +  
        weights['Efficiency'] * df_properties_scaled['Efficiency']
    )


    name = cmp.__class__.__name__

    time = df_properties[df_properties['scheme_name']==name]['Time'].values[0]
    acc = df_properties[df_properties['scheme_name']==name]['Acc'].values[0]
    eff = df_properties[df_properties['scheme_name']==name]['Efficiency'].values[0]

    combined = df_properties_scaled[df_properties_scaled['scheme_name']==name]['Combined_Score'].values[0]


    return name, acc, time, eff, combined
    
    
