import torch
import torch.nn as nn
from mean_std import mean, std


Args = {"device" : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        #"Act_fu" : nn.ReLU(),
        "mean": mean / 255,
        "std": std / 255,
        "lr" : 0.001,
        "eta_min" : 0.00001,
        #"milestone": [10, 60, 90],
        "weight_decay" : 0.0001,
        "batch_size": 16,
        "Epoch" : 250,
        #"decay_epoch" : [32000, 48000],
        #"num_fold" : 5,
        "patience" : 25
        }


'''
# cifar10

lr : 0.001,
eta_min : 0.00001,
weight_decay = 0.001,
batch_size = 64,
epoch = 250
optimizer = sgd,
scheduler = cosinewarmuprestart

top1 acc = 85~86%

'''

