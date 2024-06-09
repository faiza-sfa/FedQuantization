# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:41:09 2024

@author: Faiza
"""

from abc import ABC, abstractmethod
import numpy as np
import struct
from typing import Tuple
from bitarray import bitarray
# from src.compressors.compressor import Compressor

import pandas as pd

import torch
from tqdm import tqdm
from load_dataset import Dataset
import os

import time
# import matplotlib.pyplot as plt
# from PIL import Image
import cv2
import sys
from datetime import datetime


import math

import struct

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.eval("using Random; Random.seed!(0)")
Main.include("src/compressors/qsgd.jl")

from bitarray import bitarray
from bitarray.util import int2ba, ba2int



#%%

from abc import ABC, abstractmethod
import numpy as np

class Compressor(ABC):
    @abstractmethod
    def encode(self, array: np.ndarray) -> bytes:
        pass
    @abstractmethod
    def decode(self, array: bytes) -> np.ndarray:
        pass

#%%SETTING UP Julia and qsgd32 Class




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
#%% testing qsgd quantization
arr = list(np.random.rand(1,16).flatten()*10)
arr = np.array(arr, dtype = np.float32)
print("Before Quantization: ", arr)
cmp = QSGD(2,True)
enc = cmp.encode(arr)
print("Encoded",enc)
dec = cmp.decode(enc)
print("After Quantization: ", dec)

print(f"{100 * (sys.getsizeof(dec)-sys.getsizeof(enc))/sys.getsizeof(dec)}% smaller ")


#%%ETTING UP Julia and gzip Class

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


#%% testing gzip quantization
arr = list(np.random.rand(1,16).flatten()*10)
arr = np.array(arr, dtype = np.float32)
print("Before Quantization: ", arr)
cmp = GZip(1)
enc = cmp.encode(arr)
print("Encoded",enc)
dec = cmp.decode(enc)
print("After Quantization: ", dec)

print(f"{100 * (sys.getsizeof(dec)-sys.getsizeof(enc))/sys.getsizeof(dec)}% smaller ")



#%%FP8

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
#%% testing fp32 quantization

arr = list(np.random.rand(1,16).flatten()*10)
arr = np.array(arr, dtype = np.float32)
print("Before Quantization: ", arr)
cmp = FP8()
enc = cmp.encode(arr)
print("Encoded",enc)
dec = cmp.decode(enc)
print("After Quantization: ", dec)

print(f"{100 * (sys.getsizeof(dec)-sys.getsizeof(enc))/sys.getsizeof(dec)}% smaller ")



#%% PAQ with Compression


class MNIST_PAQ_COMP:
	def __init__(self, filename="saved_models", number_of_clients=1, aggregate_epochs=10, local_epochs=5, precision=7, r=1.0):
		self.model = None
		self.criterion = torch.nn.CrossEntropyLoss()
		self.optimizer = None
		self.number_of_clients = number_of_clients
		self.aggregate_epochs = aggregate_epochs
		self.local_epochs = local_epochs
		self.precision = precision
		self.r = r
		self.filename = filename
		self.compression_ratio = 0

	def define_model(self):
		self.model = torch.nn.Sequential(
			torch.nn.Conv2d(1, 2, kernel_size=5),
			torch.nn.ReLU(),
			torch.nn.Conv2d(2, 4, kernel_size=7),
			torch.nn.ReLU(),
			torch.nn.Flatten(),
			torch.nn.Linear(1296, 512),
			torch.nn.ReLU(),
			torch.nn.Linear(512, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 32),
			torch.nn.ReLU(),
			torch.nn.Linear(32, 10),
			torch.nn.Softmax(dim=1),
		)		

	def get_weights(self, dtype=np.float32):
        
		precision = self.precision 
		weights = []
		start_size = 0
		end_size = 0
        
        
		for layer in self.model:
			try:
# 				weights.append([np.around(layer.weight.detach().numpy().astype(dtype), decimals=precision), np.around(layer.bias.detach().numpy().astype(dtype), decimals=precision)])

                # layer_name = layer.__class__name__
                # if layer_name = 'Conv2d':
                #     layer_np = layer.weight.detach().numpy().astype(dtype)
                #     for filters in layer_np:
                #         for channels in filters:
                #             for inp_dim1 in channels:
                #                 fp8.encode(inp_dim1)
                
				layer_weights = layer.weight.detach().numpy().astype(dtype)
				start_size = start_size + sys.getsizeof(layer_weights)
				layer_bias = layer.bias.detach().numpy().astype(dtype)
				layer_weights_enc = cmp.encode(layer_weights.flatten())
				layer_bias_enc = cmp.encode(layer_bias.flatten())
				decoded_weights = cmp.decode(layer_weights_enc).reshape(layer_weights.shape)
				end_size = end_size + sys.getsizeof(decoded_weights)
				decoded_bias = cmp.decode(layer_bias_enc).reshape(layer_bias.shape)
				weights.append([decoded_weights,decoded_bias])
			except:
				continue
		self.compression_ratio = ((start_size-end_size)/start_size)*100
		return np.array(weights)

	def set_weights(self, weights):
		index = 0
		for layer_no, layer in enumerate(self.model):
			try:
				_ = self.model[layer_no].weight
				self.model[layer_no].weight = torch.nn.Parameter(weights[index][0])
				self.model[layer_no].bias = torch.nn.Parameter(weights[index][1])
				index += 1
			except:
				continue

	def average_weights(self, all_weights):
		all_weights = np.array(all_weights)
		print("all_weights",all_weights)
		all_weights = np.median(all_weights, axis=0)
		all_weights = [[torch.from_numpy(i[0].astype(np.float32)), torch.from_numpy(i[1].astype(np.float32))] for i in all_weights]
		return all_weights


	def median_weights(self, all_weights):
		"""Aggregate weights using the median."""
    # Assuming all_weights is a list of lists of numpy arrays
		num_layers = len(all_weights[0])
		median_weights = []
		for i in range(num_layers):
        # Extract the same layer across all clients
			layer_weights = np.array([client[i] for client in all_weights])
        # Calculate the median across the first axis (across clients)
			layer_median = np.median(layer_weights, axis=0)
        # Convert to PyTorch tensor and append to the result list
			median_weights.append(torch.from_numpy(layer_median.astype(np.float32)))
    
		return median_weights


	def client_generator(self, train_x, train_y):
		number_of_clients = self.number_of_clients
		size = train_y.shape[0]//number_of_clients
		train_x, train_y = train_x.numpy(), train_y.numpy()
		train_x = np.array([train_x[i:i+size] for i in range(0, len(train_x)-len(train_x)%size, size)])
		train_y = np.array([train_y[i:i+size] for i in range(0, len(train_y)-len(train_y)%size, size)])
		train_x = torch.from_numpy(train_x)
		train_y = torch.from_numpy(train_y)
		return train_x, train_y

	def single_client(self, dataset, weights, E):
		self.define_model()
		if weights is not None:
			self.set_weights(weights)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
		for epoch in range(E):
			running_loss = 0
			for batch_x, target in zip(dataset['x'], dataset['y']):
				output = self.model(batch_x)
				loss = self.criterion(output, target)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				running_loss += loss.item()
			running_loss /= len(dataset['y'])
		weights = self.get_weights()
		
		return weights, running_loss
	def test_aggregated_model(self, test_x, test_y, epoch):
		acc = 0
		with torch.no_grad():
			for batch_x, batch_y in zip(test_x, test_y):
				y_pred = self.model(batch_x)
				y_pred = torch.argmax(y_pred, dim=1)
				acc += torch.sum(y_pred == batch_y)/y_pred.shape[0]
		torch.save(self.model, "./"+self.filename+"/model_epoch_"+str(epoch+1)+".pt")
		return (acc/test_x.shape[0])
			

	def train_aggregator(self, datasets, datasets_test):
		local_epochs = self.local_epochs
		aggregate_epochs = self.aggregate_epochs
		os.system('mkdir '+self.filename)
		E = local_epochs
		aggregate_weights = None
		for epoch in range(aggregate_epochs):
			all_weights = []
			client = 0
			running_loss = 0
			selections = np.arange(datasets['x'].shape[0])
			np.random.shuffle(selections)
			selections = selections[:int(self.r*datasets['x'].shape[0])]
			clients = tqdm(zip(datasets['x'][selections], datasets['y'][selections]), total=selections.shape[0])
			for dataset_x, dataset_y in clients:
				dataset = {'x':dataset_x, 'y':dataset_y}
				weights, loss = self.single_client(dataset, aggregate_weights, E)
				running_loss += loss
				all_weights.append(weights)
				client += 1
				clients.set_description(str({"Epoch":epoch+1,"Loss": round(running_loss/client, 5)}))
				clients.refresh()
			aggregate_weights = self.median_weights(all_weights)
			self.set_weights(aggregate_weights)
			test_acc = self.test_aggregated_model(datasets_test['x'], datasets_test['y'], epoch)
			print("Test Accuracy:", round(test_acc.item(), 5))
			print("Compression Ratio ", self.compression_ratio)
			clients.close()


#%%

number_of_clients = 328
aggregate_epochs = 10
local_epochs = 3
r = 0.5
current_time = datetime.now().time()
epoch_time = time.time()

filename = f"saved_models_{epoch_time}"

train_x, train_y, test_x, test_y = Dataset().load_csv()

m_8 = MNIST_PAQ_COMP(filename=filename, r=r, number_of_clients=number_of_clients, aggregate_epochs=aggregate_epochs, local_epochs=local_epochs)
train_x, train_y = m_8.client_generator(train_x, train_y)
m_8.train_aggregator({'x':train_x, 'y':train_y}, {'x':test_x, 'y':test_y})



















