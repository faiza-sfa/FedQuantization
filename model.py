import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from load_dataset import Dataset
import os
import sys
import time


class MNIST_PAQ_COMP:
	def __init__(self, cmp, filename="saved_models", number_of_clients=1, aggregate_epochs=10, local_epochs=5, precision=7, r=1.0):
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
		self.cmp=cmp

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

	def get_weights_custom(self, dtype=np.float32):
        
		precision = self.precision 
		weights = []
		start_size = 0
		end_size = 0
        
        
		for layer in self.model:			
			try:
				layer_weights = layer.weight.detach().numpy().astype(dtype)
				start_size = start_size + sys.getsizeof(layer_weights)
				# layer_weights_enc = self.cmp.encode(layer_weights)
				layer_weights_enc = self.cmp.encode(layer_weights.flatten())
				decoded_weights = self.cmp.decode(layer_weights_enc)
				decoded_weights = decoded_weights.astype(dtype)
				decoded_weights = decoded_weights.reshape(layer_weights.shape)
				end_size = end_size + sys.getsizeof(decoded_weights)
		
				layer_bias = layer.bias.detach().numpy().astype(dtype)
				# layer_bias_enc = self.cmp.encode(layer_bias)	
				layer_bias_enc = self.cmp.encode(layer_bias.flatten())			
				decoded_bias = self.cmp.decode(layer_bias_enc)
				decoded_bias = decoded_bias.astype(dtype)
				decoded_bias = decoded_bias.reshape(layer_bias.shape)
				weights.append([decoded_weights,decoded_bias])
				
			except:
				continue
		self.compression_ratio = ((start_size-end_size)/start_size)*100
		return np.array(weights),start_size,end_size



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
		# weights,start_size,end_size = self.get_weights()
		weights,start_size,end_size = self.get_weights_custom()
		
		return weights, running_loss, start_size,end_size
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
				weights, loss, start_size,end_size = self.single_client(dataset, aggregate_weights, E)
				running_loss += loss
				all_weights.append(weights)
				client += 1
				clients.set_description(str({"Epoch":epoch+1,"Loss": round(running_loss/client, 5)}))
				clients.refresh()
			aggregate_weights = self.average_weights(all_weights)
			agg_weight = self.set_weights(aggregate_weights)
			test_acc = self.test_aggregated_model(datasets_test['x'], datasets_test['y'], epoch)
			print("Test Accuracy:", round(test_acc.item(), 5))
			print("Compression Ratio ", self.compression_ratio)
			# print()
			print(f'start_size: {start_size/1024},end_size: {end_size}')
			clients.close()
		return agg_weight



class MNIST_PAQ_COMP_ADAPTIVE:
	def __init__(self, cmp_list, filename="saved_models", number_of_clients=1, aggregate_epochs=10, local_epochs=5, precision=7, r=1.0):
		
		np.random.seed(786)
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
		self.cmp = None
		self.cmp_list = cmp_list
		self.client_ids = list(range(0,number_of_clients))

		self.bw_low = 20		
		self.bw_high = 2500

		self.traffic_low = 1
		self.traffic_high = 51
		# self.bw_traffic = [[random.randint(self.bw_low, self.bw_high), random.randint(1,50)] for _ in range(number_of_clients)]
	
		bandwidths = np.random.randint(self.bw_low, self.bw_high + 1, size=self.number_of_clients)
		traffic = np.random.randint(self.traffic_low, self.traffic_high, size=self.number_of_clients)
		self.bw_traffic = list(zip(bandwidths, traffic))


#a list of tuple of shape(list) number_of_clients

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

	def get_weights_custom(self, dtype=np.float32):
        
		precision = self.precision 
		weights = []
		start_size = 0
		end_size = 0
        

        
		for layer in self.model:			
			try:
				layer_weights = layer.weight.detach().numpy().astype(dtype)
				start_size = start_size + sys.getsizeof(layer_weights)
				# layer_weights_enc = self.cmp.encode(layer_weights)
				layer_weights_enc = self.cmp.encode(layer_weights.flatten())
				decoded_weights = self.cmp.decode(layer_weights_enc)
				decoded_weights = decoded_weights.astype(dtype)
				decoded_weights = decoded_weights.reshape(layer_weights.shape)
				end_size = end_size + sys.getsizeof(decoded_weights)
		
				layer_bias = layer.bias.detach().numpy().astype(dtype)
				# layer_bias_enc = self.cmp.encode(layer_bias)	
				layer_bias_enc = self.cmp.encode(layer_bias.flatten())			
				decoded_bias = self.cmp.decode(layer_bias_enc)
				decoded_bias = decoded_bias.astype(dtype)
				decoded_bias = decoded_bias.reshape(layer_bias.shape)
				weights.append([decoded_weights,decoded_bias])
				
			except:
				continue
		self.compression_ratio = ((start_size-end_size)/start_size)*100

		weight_array = np.array(weights)
		try:
			assert(weight_array.shape == (6,2))
		except:
			print('Scheme',self.cmp)
			print('weight_array.shape',weight_array.shape)

		return np.array(weights),start_size,end_size



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




	def get_client_bw_and_traffic(self,client_id):
		bw,traffic = self.bw_traffic[client_id]
		return bw,traffic
	


	def quanization_scheme(self,bw,traffic):
		normalized_bw = bw/(self.bw_high-self.bw_low)
		normalized_traffic = traffic/(self.traffic_high-self.traffic_low) 

		client_condition = normalized_bw * 50 + normalized_traffic * 50

		data = np.arange(0,100)
		percentile_33 = np.percentile(data, 33)
		percentile_66 = np.percentile(data, 66)

		if client_condition<=percentile_33:
			selected_cmp = self.cmp_list[0]
		elif percentile_33 < client_condition <= percentile_66:
			selected_cmp = self.cmp_list[1]
		else:
			selected_cmp = self.cmp_list[2]			
		return selected_cmp
	
		

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
		# weights,start_size,end_size = self.get_weights()
		weights,start_size,end_size = self.get_weights_custom()		
		return weights, running_loss, start_size,end_size
	
	
	def test_aggregated_model(self, test_x, test_y, epoch):
		acc = 0
		with torch.no_grad():
			for batch_x, batch_y in zip(test_x, test_y):
				y_pred = self.model(batch_x)
				y_pred = torch.argmax(y_pred, dim=1)
				acc += torch.sum(y_pred == batch_y)/y_pred.shape[0]
		torch.save(self.model, "./"+self.filename+"/model_epoch_"+str(epoch+1)+".pt")
		return (acc/test_x.shape[0])
			


	def train_aggregator(self, datasets, datasets_test,logging=False):
		local_epochs = self.local_epochs
		aggregate_epochs = self.aggregate_epochs
		os.system('mkdir '+self.filename)
		E = local_epochs
		aggregate_weights = None
		client_details = dict()
		
		for epoch in range(aggregate_epochs):
			all_weights = []
			client = 0
			running_loss = 0
			selections = np.arange(datasets['x'].shape[0])
			np.random.shuffle(selections)
			selections = selections[:int(self.r*datasets['x'].shape[0])]
			clients = tqdm(zip(datasets['x'][selections], datasets['y'][selections]), total=selections.shape[0])
			for selection_index,(dataset_x, dataset_y )in enumerate(clients):

				client_id = selections[selection_index]
				bw,traffic = self.get_client_bw_and_traffic(client_id)
				self.cmp = self.quanization_scheme(bw,traffic)
				# print(self.cmp)
				dataset = {'x':dataset_x, 'y':dataset_y}
				tic = time.time()
				weights, loss, start_size,end_size = self.single_client(dataset, aggregate_weights, E)
				toc = time.time()
				latency = toc - tic
				if logging:
					client_info = {'bw': bw, 'traffic': traffic, 'latency': latency, 'compression': self.cmp.__class__.__name__}
					if client_id not in client_details:
						client_details[client_id] = client_info

				running_loss += loss
				all_weights.append(weights)
				client += 1
				clients.set_description(str({"Epoch":epoch+1,"Loss": round(running_loss/client, 5)}))				
				clients.refresh()

		
			aggregate_weights = self.average_weights(all_weights)
			agg_weight = self.set_weights(aggregate_weights)
			self.test_acc = self.test_aggregated_model(datasets_test['x'], datasets_test['y'], epoch)
			print("Test Accuracy:", round(self.test_acc.item(), 5))
			print("Compression Ratio ", self.compression_ratio)
			# print()
			print(f'start_size: {start_size/1024},end_size: {end_size}')
			clients.close()
		if logging:
			column_names = ['Client','Bandwidth','Traffic','Latency','Compression']
			# df = pd.DataFrame.from_dict(client_details, orient='index',columns=column_names)			
			return agg_weight, client_details
			# df.to_csv(r'D:\MS_Thesis\Pre-defense-june\Results_for_update\client_latency_data.csv', index=True)
		# print("Client Details: ",client_details)

		return agg_weight
	
	def get_acc(self):
		return self.test_acc

	