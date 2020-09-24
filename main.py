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

write_name = '_' + cfg.build_net + '_' + cfg.base_net + '_' + str(cfg.k_shots) + '_Num_scene:' + str(cfg.scene_name) + '_' + str(cfg.cnn)+ '_' + str(cfg.lr) + '_' + str(cfg.act) + '_' + str(cfg.hid_dim)


writer = SummaryWriter(comment= write_name)

builder = Builder(base_model, cuda, Tensor, data)

best_val_acc = 0.0


with tqdm.tqdm(total=cfg.total_train_batches) as pbar_e:
       
    if cfg.build_net == 'matching_cz':
        for e in range(cfg.total_epochs):        
            
            total_c_loss, total_accuracy = builder.run_tuning_epoch(cfg.total_train_batches, 'train')
            print("Epoch {}: ft_loss:{} ft_accuracy:{}".format(e, total_c_loss, total_accuracy))
            writer.add_scalar('ft_epoch/Acc', total_accuracy , e) #all batch
            writer.add_scalar('ft_epoch/Loss', total_c_loss, e) #all batch            
            
            total_c_loss, total_accuracy = builder.run_training_epoch(cfg.total_train_batches)
            print("Epoch {}: train_loss:{} train_accuracy:{}".format(e, total_c_loss, total_accuracy))
            writer.add_scalar('train_epoch/Acc', total_accuracy , e) #all batch
            writer.add_scalar('train_epoch/Loss', total_c_loss, e) #all batch     
    
            total_c_loss, total_accuracy = builder.run_tuning_epoch(cfg.total_val_batches, 'val')
            total_val_c_loss, total_val_accuracy = builder.run_val_epoch(cfg.total_val_batches)
            print("Epoch {}: val_loss:{} val_accuracy:{}".format(e, total_val_c_loss, total_val_accuracy))
            writer.add_scalar('val_epoch/Acc', total_val_accuracy, e) #all batch
            writer.add_scalar('val_epoch/Loss', total_val_c_loss, e) #all batch      

           
            config_as_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('__')}
            results = {
                'config': config_as_dict,
                'weights': builder.net.state_dict(),
                'runs_dir': writer.log_dir}
            torch.save(results, 'model/'+ writer.log_dir+'.pth')
            pbar_e.update(1)
            
          
    config_as_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('__')}
    results = {
        'config': config_as_dict,
        'weights': builder.net.state_dict(),
        'runs_dir': writer.log_dir}
    torch.save(results, 'model/'+ writer.log_dir+'.pth')

