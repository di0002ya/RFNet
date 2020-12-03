##########
# Shuya 
##########

import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config as cfg



## root: data/uci folder dir 
## csv_name
## device: Phones/Wathches
## data_type: Acc/Gyro
class Explore():
    def __init__(self,root,csv_name,device,data_type):
        self.root = root
        self.csv_name = csv_name 
        self.device = device
        self.data_type = data_type
        self.df = pd.read_csv(root + csv_name)
        self.selected_col = ['User', 'Model', 'Device']
        self.selected_df = self.df[self.selected_col]
        self.scenarios = self.selected_df.drop_duplicates()     
        print(self.device + '-' + self.data_type + ':' + str(self.scenarios.shape[0]))
        
        
        
        
# Normalization Block
def block(name, in_feat, out_feat):    
#     if name == 'IN':
#         layers = [nn.Linear(in_feat, out_feat)]
#         layers.append(nn.InstanceNorm1d(cfg.win_len, 0.8))
#     elif name == 'SN':
#         layers = [SpectralNorm(nn.Linear(in_feat, out_feat))]
        
    if name == 'BN':
        # INPUT: N*win_len*in_feat
        layers = [nn.Linear(in_feat, out_feat)]
        layers.append(nn.BatchNorm1d(cfg.win_len, 0.8))
        
#     elif name == 'ADAIN':
#         layers = [nn.Linear(in_feat, out_feat)]
#         layers.append(AdaptiveInstanceNorm1d(cfg.win_len))            
    else:
        layers = [nn.Linear(in_feat, out_feat)]
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers    


# Convert int to one-hot
# def oneHotGen(tensor):
#     a = tensor.long().view(-1).numpy()
#     b = np.zeros((a.size, a.max()+1))
#     b[np.arange(a.size),a] = 1 
#     return torch.FloatTensor(b)


def print_lr(optimizer, prefix, epoch):
    all_rl = []
    for p in optimizer.param_groups:
        all_rl.append(p['lr'])
    print('{} E{:03d}:'.format(prefix, epoch), ' Learning Rate: ', set(all_rl))


def set_lr(optimizer, value):
    for p in optimizer.param_groups:
        p['lr'] = value


def decay_lr(optimizer, rate):
    for p in optimizer.param_groups:
        p['lr'] *= rate    
        
def operate_lr_for_epoch(epoch,prefix,optim):
    if epoch < len(cfg.gradual_warmup_steps) and cfg.schedule_method == 'warm_up':
        set_lr(optim, cfg.gradual_warmup_steps[epoch])
        print_lr(optim, prefix, epoch)

    elif (epoch in cfg.lr_decay_epochs) and cfg.schedule_method == 'warm_up':
        decay_lr(optim, cfg.lr_decay_rate)
        print_lr(optim, prefix, epoch)
    else:
        print_lr(optim, prefix, epoch)
        

def print_grad(named_parameters):
    """
    visualize grad
    """

    total_norm = 0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in named_parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm ** 2
            param_to_norm[n] = param_norm
            param_to_shape[n] = p.size()

    total_norm = total_norm ** (1. / 2)

    print('---Total norm {:.3f} -----------------'.format(total_norm))
    for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
            print("{:<50s}: {:.3f}, ({})".format(name, norm, param_to_shape[name]))
    print('-------------------------------', flush=True)

    return total_norm        