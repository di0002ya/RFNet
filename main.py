from data_loader import WifiDataset, FMCWDataset, UWBDataset
import tqdm
import config as cfg
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch
import torch.nn as nn
import dual as base_model     
from builder import Builder    
import dual as base_model   



if cfg.data == 'WIFI':        
    data = WifiDataset(batch_size=cfg.batch_size, 
                            scene_per_batch=cfg.scene_per_batch, 
                            k_shots = cfg.k_shots, 
                            seed=cfg.seed)
elif cfg.data == 'UWB':
    data = UWBDataset(batch_size=cfg.batch_size, 
                            scene_per_batch=cfg.scene_per_batch, 
                            k_shots = cfg.k_shots, 
                            seed=cfg.seed)    
elif cfg.data == 'FMCW':
    data = FMCWDataset(batch_size=cfg.batch_size, 
                            scene_per_batch=cfg.scene_per_batch, 
                            k_shots = cfg.k_shots, 
                            seed=cfg.seed)     
        
    
cuda = True if torch.cuda.is_available() else False 

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

builder = Builder(base_model, cuda, Tensor, data)

best_val_acc = 0.0


with tqdm.tqdm(total=cfg.total_train_batches) as pbar_e:
    
    for e in range(cfg.total_epochs):        

        total_c_loss, total_accuracy = builder.run_tuning_epoch(cfg.total_train_batches, 'train')
        print("Epoch {}: ft_loss:{} ft_accuracy:{}".format(e, total_c_loss, total_accuracy))


        total_c_loss, total_accuracy = builder.run_training_epoch(cfg.total_train_batches)
        print("Epoch {}: train_loss:{} train_accuracy:{}".format(e, total_c_loss, total_accuracy))


        total_c_loss, total_accuracy = builder.run_tuning_epoch(cfg.total_val_batches, 'val')
        
        
        total_val_c_loss, total_val_accuracy = builder.run_val_epoch(cfg.total_val_batches)
        print("Epoch {}: val_loss:{} val_accuracy:{}".format(e, total_val_c_loss, total_val_accuracy))

        pbar_e.update(1)


