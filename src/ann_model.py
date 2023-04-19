import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Sampler, Dataset
from src.data_loader import *
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import os

class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32**2, output_dim=1, num_layers=3):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        
        layers = nn.ModuleList()
        for i in range(num_layers):
            layers.extend([nn.Linear(in_features=input_dim if i==0 else hidden_dim, out_features=hidden_dim),
                          nn.ReLU()])
        self.layers = nn.Sequential(*layers)
        self.final_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        x = self.final_layer(x)
        return x    
    
def save_checkpoint(model, epoch, train_loss, val_loss, prefix=''):
    
    model_out_path = "./saves/" + prefix + '.pth'
    state = {"epoch": epoch, 
             "model": model.state_dict(),
             "train_loss": train_loss, 
             "val_loss": val_loss}
    
    if not os.path.exists("./saves/"):
        os.makedirs("./saves/")

    torch.save(state, model_out_path)
    print("model checkpoint saved @ {}".format(model_out_path))

def convert_tensor(x, y):
    x = x.to_numpy()
    y = y.to_numpy()
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y

def create_dataloader(x, y, batch_size=10):
    list_data = []
    for i in range(len(x)):
        list_data.append([x[i], y[i]])
        
    dataloader = DataLoader(list_data, batch_size, shuffle=True, drop_last=True)
    return dataloader