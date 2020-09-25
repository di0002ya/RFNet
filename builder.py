#------------
# Author: Shuya Ding
# Date: Sep 2020
#------------

import torch
import tqdm
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import config as cfg
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from matching_networks import MatchingNetwork     

class Builder:
    def __init__(self, base_model, cuda, Tensor,  data):
        self.base_model = base_model
        self.use_cuda = cuda
        self.Tensor = Tensor 
        self.total_train_batches = cfg.total_train_batches
        self.total_val_batches = cfg.total_val_batches
        self.total_test_batches = cfg.total_test_batches       
        self.data = data
        self.total_iter = 0
        self.net = MatchingNetwork(base_model,fce = cfg.fce , additional = cfg.additional_linear, use_cuda = self.use_cuda)        


        if self.use_cuda:
            cudnn.benchmark = True  # set True to speedup
            torch.cuda.manual_seed_all(2017)    
            self.net.cuda()             
                
        self.total_train_iter = 0
        self.optim = self._create_optimizer(self.net) 
        self.scheduler = ReduceLROnPlateau(self.optim, 'min',verbose=True)
    def _create_optimizer(self, model):
        # setup optimizer
        if cfg.optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        elif cfg.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, dampening=0.9)
        return optimizer

    def _adjust_learning_rate(self, optimizer):
        """
        Update the learning rate after some epochs
        :param optimizer:
        :return:
        """
        
    def run_tuning_epoch(self,total_batches, data_type):
        """
        Run the training epoch
        :param total_train_batches: Number of batches to train on
        :return:
        """
        total_c_loss = 0.0
        total_accuracy = 0.0
        if data_type == 'train':
            for n,p in self.net.named_parameters():
                p.requires_grad = True    
        else:
            for n,p in self.net.named_parameters():
                if n.split('.')[1] == 'g':
                    p.requires_grad = False
                else:
                    p.requires_grad = True             
                
        with tqdm.tqdm(total= total_batches) as pbar:
            self.net.train()
            for i in range(total_batches):
                # support_set_x: bs * (n_classes * k) * 512 * 60
                # support_set_y: bs * (n_classes * k) 
                # query_y: bs
                # query_x: bs * 512 * 60 
                if data_type =='train':
                    x_support_set, y_support_set, x_target, y_target = self.data.get_train_batch()
                elif data_type == 'val':
                    x_support_set, y_support_set, x_target, y_target = self.data.get_val_batch()                    
                elif data_type == 'test':
                    x_support_set, y_support_set, x_target, y_target = self.data.get_test_batch()                     

                
                x_support_set = Variable(torch.from_numpy(x_support_set)).float()
                y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
                x_target = Variable(torch.from_numpy(x_target)).float()
                y_target = Variable(torch.from_numpy(y_target), requires_grad=False).squeeze().long()
                
                y_support_value = y_support_set
                
                # convert to one hot encoding
                y_support_set = y_support_set.unsqueeze(2)
                sequence_length = y_support_set.size()[1]
                batch_size = y_support_set.size()[0]
                y_support_set_one_hot = Variable(
                    torch.zeros(batch_size, sequence_length, self.data.n_classes).scatter_(2,
                                                                                            y_support_set.data,
                                                                                            1), requires_grad=False)
                
                import GPUtil
                GPUtil.showUtilization()
                if  self.use_cuda:
                    if cfg.use_para:
                        preds, _  = self.net.module.g(x_support_set.view(-1,x_support_set.shape[2],x_support_set.shape[3]).cuda())
                    else:
                        preds, _  = self.net.g(x_support_set.view(-1,x_support_set.shape[2],x_support_set.shape[3]).cuda())
                else:
                    preds, _  = self.net.g(x_support_set.view(-1,x_support_set.shape[2],x_support_set.shape[3]))
                      
                    
                del x_support_set, x_target, y_target
                torch.cuda.empty_cache()
                # calculate the accuracy
                values, indices = preds.max(1)
                if self.use_cuda:
                    acc = torch.mean((indices.squeeze() == y_support_value.view(-1).cuda()).float())
                    c_loss = F.cross_entropy(preds,  y_support_value.view(-1).cuda().long())                
                else:
                    acc = torch.mean((indices.squeeze() == y_support_value.view(-1)).float())
                    c_loss = F.cross_entropy(preds,  y_support_value.view(-1).long())                        
                
                del y_support_value, indices
                import gc
                gc.collect()
                torch.cuda.empty_cache()                
                
                
                # optimize process
                self.optim.zero_grad()
                c_loss.backward()
                self.optim.step()

                iter_out = "ft_loss: {}, ft_accuracy: {}".format(c_loss.item(), acc.item())
                pbar.set_description(iter_out)
                pbar.update(1)
                total_c_loss += c_loss.item()
                total_accuracy += acc.item()

            total_c_loss = total_c_loss / total_batches
            total_accuracy = total_accuracy / total_batches
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            return total_c_loss, total_accuracy                
        
        
        
    def run_training_epoch(self, total_train_batches):
        

        """
        Run the training epoch
        :param total_train_batches: Number of batches to train on
        :return:
        """
        total_c_loss = 0.0
        total_accuracy = 0.0
        
        
        # ONLY FINE-TUNING LAST LAYERS
        for n,p in self.net.named_parameters():
            if n.split('.')[1] == 'g':
                p.requires_grad = False
            else:
                p.requires_grad = True    
  
            
        with tqdm.tqdm(total= total_train_batches) as pbar:
            self.net.train()
            for i in range(total_train_batches):
                # support_set_x: bs * (n_classes * k) * 512 * 60
                # support_set_y: bs * (n_classes * k) 
                # query_y: bs
                # query_x: bs * 512 * 60 
                
                x_support_set, y_support_set, x_target, y_target = self.data.get_train_batch()
                x_support_set = Variable(torch.from_numpy(x_support_set)).float()
                y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
                if cfg.reshape_to_scene:
                    x_target = torch.cat([torch.from_numpy(x_target)] * cfg.k_shots)
                    y_target = torch.cat([torch.from_numpy(y_target)] * cfg.k_shots)                  
                else:
                    x_target= torch.from_numpy(x_target)
                    y_target = torch.from_numpy(y_target)                    
                    
                x_target = Variable(x_target).float()
                y_target = Variable(y_target, requires_grad=False).squeeze().long()

                # convert to one hot encoding
                y_support_set = y_support_set.unsqueeze(2)
                sequence_length = y_support_set.size()[1]
                batch_size = y_support_set.size()[0]
                y_support_set_one_hot = Variable(
                torch.zeros(batch_size, sequence_length, self.data.n_classes).scatter_(2,
                                                                                        y_support_set.data,
                                                                                        1), requires_grad=False)

                if  self.use_cuda:
                    acc, c_loss = self.net(x_support_set.cuda(), y_support_set_one_hot.cuda(), x_target.cuda(), y_target.cuda())
                else:
                    acc, c_loss  = self.net(x_support_set, y_support_set_one_hot, x_target, y_target)
                     
    
                    
                del x_support_set, y_support_set_one_hot, x_target, y_target
                torch.cuda.empty_cache()
                
                
                # optimize process
                self.optim.zero_grad()
                c_loss.backward()
                self.optim.step()

                iter_out = "tr_loss: {}, tr_accuracy: {}".format(c_loss.item(), acc.item())
                pbar.set_description(iter_out)
                pbar.update(1)
                total_c_loss += c_loss.item()
                total_accuracy += acc.item()
                # self.total_train_iter+=1

            total_c_loss = total_c_loss / total_train_batches
            total_accuracy = total_accuracy / total_train_batches
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            return total_c_loss, total_accuracy        
        
    def run_val_epoch(self, total_val_batches):
        """
        Run the training epoch
        :param total_train_batches: Number of batches to train on
        :return:
        """
        total_c_loss = 0.0
        total_accuracy = 0.0
        # optimizer = self._create_optimizer(self.matchNet, self.lr)

        with tqdm.tqdm(total= total_val_batches) as pbar:
            with torch.no_grad():
                self.net.eval()
                for i in range(total_val_batches):
                    # support_set_x: bs * (n_classes * k) * 512 * 60
                    # support_set_y: bs * (n_classes * k) 
                    # query_y: bs
                    # query_x: bs * 512 * 60 

                    x_support_set, y_support_set, x_target, y_target = self.data.get_val_batch()
                    x_support_set = Variable(torch.from_numpy(x_support_set)).float()
                    y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
                    if cfg.reshape_to_scene:
                        x_target = torch.cat([torch.from_numpy(x_target)] * cfg.k_shots)
                        y_target = torch.cat([torch.from_numpy(y_target)] * cfg.k_shots)                  
                    else:
                        x_target= torch.from_numpy(x_target)
                        y_target = torch.from_numpy(y_target)
                        
                        
                    x_target = Variable(x_target).float()
                    y_target = Variable(y_target, requires_grad=False).squeeze().long()
                    y_support_set = y_support_set.unsqueeze(2)
                    sequence_length = y_support_set.size()[1]
                    batch_size = y_support_set.size()[0]
                    y_support_set_one_hot = Variable(
                    torch.zeros(batch_size, sequence_length, self.data.n_classes).scatter_(2,
                                                                                            y_support_set.data,
                                                                                            1), requires_grad=False)

                    if  self.use_cuda:
                        acc, c_loss = self.net(x_support_set.cuda(), y_support_set_one_hot.cuda(), x_target.cuda(),
                                                    y_target.cuda())
                    else:
                        acc, c_loss = self.net(x_support_set, y_support_set_one_hot, x_target, y_target)

                    del x_support_set, y_support_set_one_hot, x_target, y_target
                    torch.cuda.empty_cache()

                    iter_out = "val_loss: {}, val_accuracy: {}".format(c_loss.item(), acc.item())
                    pbar.set_description(iter_out)
                    pbar.update(1)
                    total_c_loss += c_loss.item()
                    total_accuracy += acc.item()

                total_c_loss = total_c_loss / total_val_batches
                total_accuracy = total_accuracy / total_val_batches
                self.scheduler.step(total_c_loss)
            return total_c_loss, total_accuracy   
        
    def run_test_epoch(self, total_test_batches):
        """
        Run the training epoch
        :param total_train_batches: Number of batches to train on
        :return:
        """
        total_c_loss = 0.0
        total_accuracy = 0.0

        with tqdm.tqdm(total= total_test_batches) as pbar:
            with torch.no_grad():
                self.net.eval()
                for i in range(total_test_batches):


                    x_support_set, y_support_set, x_target, y_target = self.data.get_test_batch()
                    x_support_set = Variable(torch.from_numpy(x_support_set)).float()
                    y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
                    if cfg.reshape_to_scene:
                        x_target = torch.cat([torch.from_numpy(x_target)] * cfg.k_shots)
                        y_target = torch.cat([torch.from_numpy(y_target)] * cfg.k_shots) 
                    else:
                        x_target= torch.from_numpy(x_target)
                        y_target = torch.from_numpy(y_target)

                    x_target = Variable(x_target).float()
                    y_target = Variable(y_target, requires_grad=False).squeeze().long()

                    # convert to one hot encoding
                    y_support_set = y_support_set.unsqueeze(2)
                    sequence_length = y_support_set.size()[1]
                    batch_size = y_support_set.size()[0]
                    y_support_set_one_hot = Variable(
                    torch.zeros(batch_size, sequence_length, self.data.n_classes).scatter_(2,
                                                                                            y_support_set.data,
                                                                                            1), requires_grad=False)

                    if  self.use_cuda:
                        acc, c_loss = self.net(x_support_set.cuda(), y_support_set_one_hot.cuda(), x_target.cuda(),
                                                    y_target.cuda())
                    else:
                        acc, c_loss = self.net(x_support_set, y_support_set_one_hot, x_target, y_target)

                    del x_support_set, y_support_set_one_hot, x_target, y_target
                    torch.cuda.empty_cache()

                    iter_out = "test_loss: {}, test_accuracy: {}".format(c_loss.item(), acc.item())
                    pbar.set_description(iter_out)
                    pbar.update(1)
                    total_c_loss += c_loss.item()
                    total_accuracy += acc.item()

                total_c_loss = total_c_loss / total_test_batches
                total_accuracy = total_accuracy / total_test_batches
            return total_c_loss, total_accuracy           
        
        