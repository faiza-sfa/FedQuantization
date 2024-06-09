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
from src.compressors.compressor import Compressor

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


# import math
# from src.compressors.compressor import Compressor
# import struct

# from julia.api import Julia
# jl = Julia(compiled_modules=False)
# from julia import Main
# Main.eval("using Random; Random.seed!(0)")
# Main.include("src/compressors/qsgd.jl")

# from bitarray import bitarray
# from bitarray.util import int2ba, ba2int


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


    #%% my quantization

import cv2


class customCompression():
    def __init__(self, quantization_level: int = 50):
        self.quantization_level = quantization_level
        #quantization_level from 0 to 100
        


    def encode(self,data):
        
        self.xp = [data.min(), data.max()]
        self.fp = [0, 255]
        img = np.interp(data, self.xp, self.fp)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quantization_level] 
        result, encimg = cv2.imencode('.jpg', img, encode_param)

        return encimg

    def decode(self, encimg) :
        decimg = cv2.cvtColor(cv2.imdecode(encimg, 1), cv2.COLOR_BGR2GRAY) 
        decimg = np.interp(decimg, self.fp, self.xp)
        return decimg

#%%

data = np.random.rand(50, 50)
cmp = customCompression()
enc = cmp.encode(data) #quantization_level from 0 to 100
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
# 				print("layer_weights shape",layer_weights.shape)
				start_size = start_size + sys.getsizeof(layer_weights)
				layer_bias = layer.bias.detach().numpy().astype(dtype)
				layer_weights_enc = cmp.encode(layer_weights.flatten())
# 				print("layer_weights_enc shape",layer_weights_enc.shape)
				layer_bias_enc = cmp.encode(layer_bias.flatten())
				decoded_weights = cmp.decode(layer_weights_enc).reshape(layer_weights.shape)
# 				print("layer_weights_dec shape",decoded_weights.shape)
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
		all_weights = np.mean(all_weights, axis=0)
		all_weights = [[torch.from_numpy(i[0].astype(np.float32)), torch.from_numpy(i[1].astype(np.float32))] for i in all_weights]
		return all_weights

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
			aggregate_weights = self.average_weights(all_weights)
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
tic = time.time()
m_8 = MNIST_PAQ_COMP(filename=filename, r=r, number_of_clients=number_of_clients, aggregate_epochs=aggregate_epochs, local_epochs=local_epochs)
train_x, train_y = m_8.client_generator(train_x, train_y)
m_8.train_aggregator({'x':train_x, 'y':train_y}, {'x':test_x, 'y':test_y})
print("Time taken: ", time.time()- tic)
print(f"{100 * (sys.getsizeof(dec)-sys.getsizeof(enc))/sys.getsizeof(dec)}% smaller ")
print(f'Quantization: {cmp}')
#%%

number_of_clients = 328
aggregate_epochs = 10
local_epochs = 3
r = 0.5
# current_time = now().time()
epoch_time = time.time()

filename = f"saved_models_{epoch_time}"

train_x, train_y, test_x, test_y = Dataset().load_csv()
tic = time.time()
m_8 = MNIST_PAQ_COMP(filename=filename, r=r, number_of_clients=number_of_clients, aggregate_epochs=aggregate_epochs, local_epochs=local_epochs)
train_x, train_y = m_8.client_generator(train_x, train_y)
m_8.train_aggregator({'x':train_x, 'y':train_y}, {'x':test_x, 'y':test_y})
print("Time taken: ", time.time()- tic)
print(f"{100 * (sys.getsizeof(dec)-sys.getsizeof(enc))/sys.getsizeof(dec)}% smaller ")
print(f'Quantization: {cmp}')


#%%

class Test:
	def __init__(self, train_x, train_y, test_x, test_y):
		self.train_x, self.train_y, self.test_x, self.test_y = train_x, train_y, test_x, test_y
		self.criterion = torch.nn.CrossEntropyLoss()
		
	def single_model(self, path):
		model = torch.load(path)
		model.eval()

		with torch.no_grad():

			train_loss, train_acc = 0, 0

			for batch_x, target in zip(self.train_x, self.train_y):
				output = model(batch_x)
				loss = self.criterion(output, target)
				output = torch.argmax(output, dim=1)
				acc = torch.sum(output == target)/target.shape[0]
				loss = loss.item()
				acc = acc.item()
				train_loss += loss
				train_acc += acc

			train_loss, train_acc = train_loss/self.train_x.shape[0], train_acc/self.train_x.shape[0]

			test_loss, test_acc = 0, 0

			for batch_x, target in zip(self.test_x, self.test_y):
				output = model(batch_x)
				loss = self.criterion(output, target)
				output = torch.argmax(output, dim=1)
				acc = torch.sum(output == target)/target.shape[0]
				loss = loss.item()
				acc = acc.item()
				test_loss += loss
				test_acc += acc

		test_loss, test_acc = test_loss/self.test_x.shape[0], test_acc/self.test_x.shape[0]

		return [train_loss, train_acc, test_loss, test_acc]

	def analyse_type(self, filename):
		model_names = os.listdir(filename)
		models = np.array([filename+i for i in model_names])
		model_names_index = np.argsort(np.array([int(i[len('model_epoch_'):-3]) for i in model_names]))
		models = models[model_names_index]
		performance = []
		for model in models:
			performance.append(self.single_model(model))
		performance = pd.DataFrame(np.array(performance))
		performance.columns = ['train_loss', 'train_acc', 'test_loss', 'test_acc']
		performance.to_csv('./results/performance_metrics/'+filename[len('./results/models/'):-1]+'.csv', index=False)

		performance = performance.values
		best_index = np.argmin(performance.T[2])
		return performance[best_index]

	def analyse_all(self):
		bar = tqdm(total=3*4*3)
		all_best_performances = []
		with bar:
			for local_epochs in [1, 3, 5]:
				for r in [0.5, 0.667, 0.833, 1.0]:
					for precision in [5, 6, 7]:
						filename = "./results/models/saved_models_local_epochs_"+str(local_epochs)+"_r_"+str(r).replace('.', '_')+"_precision_"+str(precision)+"/"
						best = self.analyse_type(filename)
						best = [local_epochs, r, precision] + [i for i in best]
						all_best_performances.append(best)
						bar.update(1)
		all_best_performances = pd.DataFrame(np.array(all_best_performances))
		all_best_performances.columns = ['local_epochs', 'r', 'precision', 'train_loss', 'train_acc', 'test_loss', 'test_acc']
		all_best_performances.to_csv('./results/best_performances.csv', index=False)

	def image_beautifier(self):

		image_names = sorted(['./results/'+i for i in os.listdir('./results/') if '.png' in i])
		for names in [image_names[i:i+4] for i in range(0, 12, 4)]:
			images = [Image.open(x) for x in names]
			widths, heights = zip(*(i.size for i in images))

			total_width = sum(widths)
			max_height = max(heights)

			new_im = Image.new('RGB', (total_width, max_height))

			x_offset = 0
			for im in images:
				new_im.paste(im, (x_offset,0))
				x_offset += im.size[0]

			name = names[0][len('./results/'):names[0].index('__')]
			new_im.save(name+'_variations.png')

		### Resizing for actual use

		for image in [i for i in os.listdir() if '_variations.png' in i]:
			img = cv2.resize(cv2.imread(image), (1280, 240))
			cv2.imwrite(image, img)

	def image(self, performances, names, pic_name):
		for i,name in enumerate(['train_loss', 'train_acc', 'test_loss', 'test_acc']):
			plt.cla()
			for j,performance in enumerate(performances):
				plt.plot(np.arange(10), performance.T[i], label=names[j])
			plt.legend()
			if i%2==1:
				plt.ylim([-0.01, 1.01])
			plt.title(name+' - '+pic_name)
			plt.savefig('./results/'+pic_name+'__'+name+'_analysis.png')

	def image_generator(self):
		performance = pd.read_csv('./results/best_performances.csv')
		features = performance.columns
		performance = performance.values
		best_index = np.argmin(performance.T[-2])
		print("Best Performance By: ", {i:j for i,j in zip(features, performance[best_index])})

		local_epochs, r, precision = performance[best_index][:3]
		local_epochs = int(local_epochs)
		precision = int(precision)

		### Local Epochs
		print("Analysing local_epochs with r and precision fixed to", r, precision, "respectively.")
		performances = []
		for local_epoch in [1, 3, 5]:
			filename = "./results/performance_metrics/saved_models_local_epochs_"+str(local_epoch)+"_r_"+str(r).replace('.', '_')+"_precision_"+str(precision)+".csv"
			performance = pd.read_csv(filename)
			performance = performance.values
			performances.append(performance)
		performances = np.array(performances)
		self.image(performances, names=['local_epochs='+str(i) for i in [1, 3, 5]], pic_name='local_epochs')

		### r
		print("Analysing local_epochs with local_epochs and precision fixed to", local_epochs, precision, "respectively.")
		performances = []
		for r_id in [0.5, 0.667, 0.833, 1.0]:
			filename = "./results/performance_metrics/saved_models_local_epochs_"+str(local_epochs)+"_r_"+str(r_id).replace('.', '_')+"_precision_"+str(precision)+".csv"
			performance = pd.read_csv(filename)
			performance = performance.values
			performances.append(performance)
		performances = np.array(performances)
		self.image(performances, names=['r='+str(i) for i in [0.5, 0.667, 0.833, 1.0]], pic_name='r')

		### r
		print("Analysing local_epochs with local_epochs and r fixed to", local_epochs, r, "respectively.")
		performances = []
		for p in [5, 6, 7]:
			filename = "./results/performance_metrics/saved_models_local_epochs_"+str(local_epochs)+"_r_"+str(r).replace('.', '_')+"_precision_"+str(p)+".csv"
			performance = pd.read_csv(filename)
			performance = performance.values
			performances.append(performance)
		performances = np.array(performances)
		self.image(performances, names=['precision='+str(i) for i in [5, 6, 7]], pic_name='precision')

		self.image_beautifier()






















