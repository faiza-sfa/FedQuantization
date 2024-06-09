# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:49:37 2024


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

#%%

def client_generator(train_x, train_y, number_of_clients):
    number_of_clients = number_of_clients
    size = train_y.shape[0]//number_of_clients
    train_x, train_y = train_x.numpy(), train_y.numpy()
    train_x = np.array([train_x[i:i+size] for i in range(0, len(train_x)-len(train_x)%size, size)])
    train_y = np.array([train_y[i:i+size] for i in range(0, len(train_y)-len(train_y)%size, size)])
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    return train_x, train_y


criterion = torch.nn.CrossEntropyLoss()
def define_model():
        model = torch.nn.Sequential(
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
        torch.nn.Softmax(dim=1),)
        return model
        
def set_weights(model,weights):
    index = 0
    for layer_no, layer in enumerate(model):
        try:
            _ = model[layer_no].weight
            model[layer_no].weight = torch.nn.Parameter(weights[index][0])
            model[layer_no].bias = torch.nn.Parameter(weights[index][1])
            index += 1
        except:
            continue

def get_weights(model,dtype=np.float32):
    precision = 7 
    weights = []
    for layer in model:
        try:
            weights.append([np.around(layer.weight.detach().numpy().astype(dtype), decimals=precision), np.around(layer.bias.detach().numpy().astype(dtype), decimals=precision)])
        except:
            continue
    return np.array(weights)


def single_client( dataset, weights, E):
    model = define_model()
    if weights is not None:
        set_weights(model, weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(E):
        running_loss = 0
        for batch_x, target in zip(dataset['x'], dataset['y']):
            output = model(batch_x)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss /= len(dataset['y'])
    weights = get_weights(model)
    
    return weights, running_loss, model


def average_weights( all_weights):
    all_weights = np.array(all_weights)
    all_weights = np.mean(all_weights, axis=0)
    all_weights = [[torch.from_numpy(i[0].astype(np.float32)), torch.from_numpy(i[1].astype(np.float32))] for i in all_weights]
    return all_weights


def test_aggregated_model(model, test_x, test_y, epoch):
    acc = 0
    with torch.no_grad():
        for batch_x, batch_y in zip(test_x, test_y):
            y_pred = model(batch_x)
            y_pred = torch.argmax(y_pred, dim=1)
            acc += torch.sum(y_pred == batch_y)/y_pred.shape[0]
    torch.save(model, "./"+filename+"/model_epoch_"+str(epoch+1)+".pt")
    return (acc/test_x.shape[0])


number_of_clients = 328
aggregate_epochs = 10
local_epochs = 3
r = 0.5
filename = "saved_models"


train_x, train_y, test_x, test_y = Dataset().load_csv()
datasets = {'x':train_x, 'y':train_y} 
datasets_test =  {'x':test_x, 'y':test_y}
# m_8 = MNIST_PAQ_FP8(filename=filename, r=r, number_of_clients=number_of_clients, aggregate_epochs=aggregate_epochs, local_epochs=local_epochs)
train_x, train_y = client_generator(train_x, train_y,number_of_clients)


local_epochs = local_epochs


aggregate_epochs = aggregate_epochs

E = local_epochs
aggregate_weights = None
if not os.path.exists(filename):
    os.makedirs(filename)
for epoch in range(aggregate_epochs):
    all_weights = []
    client = 0
    running_loss = 0
    selections = np.arange(datasets['x'].shape[0])
    np.random.shuffle(selections)
    selections = selections[:int(r*datasets['x'].shape[0])]
    clients = tqdm(zip(datasets['x'][selections], datasets['y'][selections]), total=selections.shape[0])

    for dataset_x, dataset_y in clients:
        dataset = {'x':dataset_x, 'y':dataset_y}
        weights, loss, model = single_client(dataset, aggregate_weights, E)
        running_loss += loss
        all_weights.append(weights)
        client += 1
        clients.set_description(str({"Epoch":epoch+1,"Loss": round(running_loss/client, 5)}))
        clients.refresh()
    aggregate_weights = average_weights(all_weights)
    set_weights(model, aggregate_weights)
    test_acc = test_aggregated_model(model,datasets_test['x'], datasets_test['y'], epoch)
    print("Test Accuracy:", round(test_acc.item(), 5))
    clients.close()
