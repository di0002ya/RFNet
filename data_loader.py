# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Ding Shuya
# Copyright (c) 2019

# @FILE    :data_loader.py
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import config as cfg
import torch
from torch.utils.data import Dataset, DataLoader
import math
class WifiDataset():
    def __init__(self, 
                 batch_size = cfg.batch_size, 
                 scene_per_batch=cfg.scene_per_batch, 
                 k_shots = cfg.k_shots, 
                 seed= cfg.seed):
        """
        Construct N-shot dataset
        :param batch_size:  Experiment batch_size
        :param scene_per_batch: Integer indicating the number of scenes per batch
        :param k_shots: Integer indicating support samples per class
        :param seed: seed for random function
        :param shuffle: if shuffle the dataset
        """
        self.k_shots = k_shots
        self.scene_per_batch = scene_per_batch
        self.batch_size = batch_size 
        
        
        np.random.seed(seed)
        self.x = torch.load('data/X_'+str(cfg.scene_name)+'_scenarios.pth')
        self.y = torch.load('data/Y_'+str(cfg.scene_name)+'_scenarios.pth')        
        ratio_len = np.array([0.6,0.2,0.2]) * self.x.shape[0]

        self.x_train, self.x_val, self.x_test = self.x[:int(ratio_len[0])], self.x[int(ratio_len[0]):int(ratio_len[0])+int(ratio_len[1])], self.x[int(ratio_len[0])+int(ratio_len[1]):]
        self.y_train, self.y_val, self.y_test = self.y[:int(ratio_len[0])], self.y[int(ratio_len[0]):int(ratio_len[0])+int(ratio_len[1])], self.y[int(ratio_len[0])+int(ratio_len[1]):]        

        self.x_train = self.processes_batch(self.x_train, torch.mean(self.x_train), torch.std(self.x_train))
        self.x_test = self.processes_batch(self.x_test, torch.mean(self.x_test), torch.std(self.x_test))
        self.x_val = self.processes_batch(self.x_val, torch.mean(self.x_val), torch.std(self.x_val))

        self.batch_size = batch_size
        self.n_scenes = self.x.shape[0]
        self.n_classes = self.x.shape[1]
        self.shots_per_class = self.x.shape[2]
        
        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.datatset = {"train": self.x_train, "val": self.x_val, "test": self.x_test}
        self.datatset_y = {"train": self.y_train, "val": self.y_val, "test": self.y_test}

    def processes_batch(self, x_batch, mean, std):
        """
        Normalizes a batch images
        :param x_batch: a batch images
        :return: normalized images
        """
        return (x_batch - mean) / std

    def _sample_new_batch(self, data_pack, data_pack_y):
        """
        Collect 1000 batches data for N-shot learning
        :param data_pack: one of(train,test,val) dataset shape[n_scenes, n_classes, shots_per_class,512,60]
        :param data_pack_y: one of(train,test,val) dataset shape[n_scenes, n_classes, shots_per_class, 6]        
        :return: A list with [support_set_x,support_set_y,target_x,target_y] ready to be fed to our networks
        """
        support_set_x = np.zeros((self.batch_size, self.n_classes, self.k_shots, data_pack.shape[3],
                                  data_pack.shape[4]), np.float32)
        support_set_y = np.zeros((self.batch_size, self.n_classes, self.k_shots, self.n_classes), np.int32)
        
        query_x = np.zeros((self.batch_size, data_pack.shape[3], data_pack.shape[4]), np.float32)
        query_y = np.zeros((self.batch_size, self.n_classes), np.int32)

        for i in range(self.batch_size):
            scene_idx = np.arange(data_pack.shape[0])
            shots_idx = np.arange(data_pack.shape[1])
            class_idx = np.arange(self.n_classes)
            choose_scene = np.random.choice(scene_idx, size=1, replace=False)
            choose_class = np.random.choice(class_idx, size=1)
            choose_samples = np.random.choice(shots_idx, size=self.k_shots + 1, replace=False)

            x_temp = data_pack[choose_scene].squeeze(0)
            x_temp = x_temp[:, choose_samples] # n_classes * (k+1) * 512 * 60 
            y_temp = data_pack_y[choose_scene].squeeze(0)
            y_temp = y_temp[:, choose_samples] # n_classes * (k+1) * 6 
            
            # split support & query 
            support_set_x[i] = x_temp[:,:-1]
            support_set_y[i] = y_temp[:,:-1]
            
            query_x[i] = x_temp[choose_class,-1] # select classes and last shots 
            query_y[i] = y_temp[choose_class,-1] # select classes and last shots 

        # support_set_x: bs * n_classes * k * 512 * 60
        # support_set_y: bs * n_classes * k * 6
        # query_x: bs * 512 * 60 
        # query_y: bs * 6 
        return support_set_x, support_set_y, query_x, query_y


    def _get_batch(self, dataset_name):
        """
        Get next batch from the dataset with name.
        :param dataset_name: The name of dataset(one of "train","val","test")
        :param augment: if rotate the images
        :return: a batch images
        """
  
        support_set_x, support_set_y, query_x, query_y = self._sample_new_batch(self.datatset[dataset_name],
                                                                                self.datatset_y[dataset_name],)

        
        if cfg.reshape_to_scene == True:
            # support_set_x: bs * n_classes * k * 512 * 60 
            support_set_x = torch.from_numpy(support_set_x).permute(0,2,1,3,4).numpy()
            # support_set_x: (bs *  k) * n_classes * 512 * 60 
            support_set_x = support_set_x.reshape(support_set_x.shape[0]*support_set_x.shape[1],
                                                  support_set_x.shape[2],
                                                  support_set_x.shape[3],
                                                  support_set_x.shape[4])

            support_set_y = torch.from_numpy(support_set_y).permute(0,2,1,3).numpy()
            support_set_y = support_set_y.reshape(support_set_y.shape[0]*support_set_y.shape[1],
                                                  support_set_y.shape[2],
                                                  support_set_y.shape[3])
            support_set_y = np.array(torch.max(torch.Tensor(support_set_y), dim = 2)[1])

            query_y = np.array(torch.max(torch.Tensor(query_y),dim = 1)[1])            
            
        else:  
            support_set_x = support_set_x.reshape(support_set_x.shape[0],
                                                  support_set_x.shape[1]*support_set_x.shape[2],
                                                  support_set_x.shape[3],
                                                  support_set_x.shape[4])

            support_set_y = support_set_y.reshape(support_set_y.shape[0],
                                                  support_set_y.shape[1]*support_set_y.shape[2],
                                                  support_set_y.shape[3])
            support_set_y = np.array(torch.max(torch.Tensor(support_set_y), dim = 2)[1])

            query_y = np.array(torch.max(torch.Tensor(query_y),dim = 1)[1])

        
        return support_set_x, support_set_y, query_x, query_y

    def get_train_batch(self):
        return self._get_batch("train")

    def get_val_batch(self):
        return self._get_batch("val")

    def get_test_batch(self):
        return self._get_batch("test")

import numpy as np
import config as cfg
import torch

class UWBDataset():
    def __init__(self, 
                 batch_size = cfg.batch_size, 
                 scene_per_batch=cfg.scene_per_batch, 
                 k_shots = cfg.k_shots, 
                 seed= cfg.seed):
        """
        Construct N-shot dataset
        :param batch_size:  Experiment batch_size
        :param scene_per_batch: Integer indicating the number of scenes per batch
        :param k_shots: Integer indicating support samples per class
        :param seed: seed for random function
        :param shuffle: if shuffle the dataset
        """
        self.k_shots = k_shots
        self.scene_per_batch = scene_per_batch
        self.batch_size = batch_size 
        #win15Noise
        
        np.random.seed(seed)
        self.x = torch.load('data/X_'+str(cfg.scene_name)+'_scenarios_UWBrea.pth')
        self.y = torch.load('data/Y_'+str(cfg.scene_name)+'_scenarios_UWBrea.pth')        
        ratio_len = np.array([0.6,0.2,0.2]) * self.x.shape[0]

        self.x_train, self.x_val, self.x_test = self.x[:int(ratio_len[0])], self.x[int(ratio_len[0]):int(ratio_len[0])+int(ratio_len[1])], self.x[int(ratio_len[0])+int(ratio_len[1]):]
        self.y_train, self.y_val, self.y_test = self.y[:int(ratio_len[0])], self.y[int(ratio_len[0]):int(ratio_len[0])+int(ratio_len[1])], self.y[int(ratio_len[0])+int(ratio_len[1]):]        

        self.x_train = self.processes_batch(self.x_train, torch.mean(self.x_train), torch.std(self.x_train))
        self.x_test = self.processes_batch(self.x_test, torch.mean(self.x_test), torch.std(self.x_test))
        self.x_val = self.processes_batch(self.x_val, torch.mean(self.x_val), torch.std(self.x_val))

        self.batch_size = batch_size
        self.n_scenes = self.x.shape[0]
        self.n_classes = self.x.shape[1]
        self.shots_per_class = self.x.shape[2]
        
        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.datatset = {"train": self.x_train, "val": self.x_val, "test": self.x_test}
        self.datatset_y = {"train": self.y_train, "val": self.y_val, "test": self.y_test}

    def processes_batch(self, x_batch, mean, std):
        """
        Normalizes a batch images
        :param x_batch: a batch images
        :return: normalized images
        """
        return (x_batch - mean) / std

    def _sample_new_batch(self, data_pack, data_pack_y):
        """
        Collect 1000 batches data for N-shot learning
        :param data_pack: one of(train,test,val) dataset shape[n_scenes, n_classes, shots_per_class,512,60]
        :param data_pack_y: one of(train,test,val) dataset shape[n_scenes, n_classes, shots_per_class, 6]        
        :return: A list with [support_set_x,support_set_y,target_x,target_y] ready to be fed to our networks
        """
        support_set_x = np.zeros((self.batch_size, self.n_classes, self.k_shots, data_pack.shape[3],
                                  data_pack.shape[4]), np.float32)
        support_set_y = np.zeros((self.batch_size, self.n_classes, self.k_shots, self.n_classes), np.int32)
        
        query_x = np.zeros((self.batch_size, data_pack.shape[3], data_pack.shape[4]), np.float32)
        query_y = np.zeros((self.batch_size, self.n_classes), np.int32)

        for i in range(self.batch_size):
            scene_idx = np.arange(data_pack.shape[0])
            shots_idx = np.arange(data_pack.shape[1])
            class_idx = np.arange(self.n_classes)
            choose_scene = np.random.choice(scene_idx, size=1, replace=False)
            choose_class = np.random.choice(class_idx, size=1)
            choose_samples = np.random.choice(shots_idx, size=self.k_shots + 1, replace=False)

            x_temp = data_pack[choose_scene].squeeze(0)
            x_temp = x_temp[:, choose_samples] # n_classes * (k+1) * 512 * 60 
            y_temp = data_pack_y[choose_scene].squeeze(0)
            y_temp = y_temp[:, choose_samples] # n_classes * (k+1) * 6 
            
            # split support & query 
            support_set_x[i] = x_temp[:,:-1]
            support_set_y[i] = y_temp[:,:-1]
            
            query_x[i] = x_temp[choose_class,-1] # select classes and last shots 
            query_y[i] = y_temp[choose_class,-1] # select classes and last shots 

        # support_set_x: bs * n_classes * k * 512 * 60
        # support_set_y: bs * n_classes * k * 6
        # query_x: bs * 512 * 60 
        # query_y: bs * 6 
        return support_set_x, support_set_y, query_x, query_y


    def _get_batch(self, dataset_name):
        """
        Get next batch from the dataset with name.
        :param dataset_name: The name of dataset(one of "train","val","test")
        :param augment: if rotate the images
        :return: a batch images
        """
  
        support_set_x, support_set_y, query_x, query_y = self._sample_new_batch(self.datatset[dataset_name],
                                                                                self.datatset_y[dataset_name],)

        
        if cfg.reshape_to_scene == True:
            # support_set_x: bs * n_classes * k * 512 * 60 
            support_set_x = torch.from_numpy(support_set_x).permute(0,2,1,3,4).numpy()
            # support_set_x: (bs *  k) * n_classes * 512 * 60 
            support_set_x = support_set_x.reshape(support_set_x.shape[0]*support_set_x.shape[1],
                                                  support_set_x.shape[2],
                                                  support_set_x.shape[3],
                                                  support_set_x.shape[4])

            support_set_y = torch.from_numpy(support_set_y).permute(0,2,1,3).numpy()
            support_set_y = support_set_y.reshape(support_set_y.shape[0]*support_set_y.shape[1],
                                                  support_set_y.shape[2],
                                                  support_set_y.shape[3])
            support_set_y = np.array(torch.max(torch.Tensor(support_set_y), dim = 2)[1])

            query_y = np.array(torch.max(torch.Tensor(query_y),dim = 1)[1])            
            
        else:  
            support_set_x = support_set_x.reshape(support_set_x.shape[0],
                                                  support_set_x.shape[1]*support_set_x.shape[2],
                                                  support_set_x.shape[3],
                                                  support_set_x.shape[4])

            support_set_y = support_set_y.reshape(support_set_y.shape[0],
                                                  support_set_y.shape[1]*support_set_y.shape[2],
                                                  support_set_y.shape[3])
            support_set_y = np.array(torch.max(torch.Tensor(support_set_y), dim = 2)[1])

            query_y = np.array(torch.max(torch.Tensor(query_y),dim = 1)[1])

        
        return support_set_x, support_set_y, query_x, query_y

    def get_train_batch(self):
        return self._get_batch("train")

    def get_val_batch(self):
        return self._get_batch("val")

    def get_test_batch(self):
        return self._get_batch("test")

class FMCWDataset():
    def __init__(self, 
                 batch_size = cfg.batch_size, 
                 scene_per_batch=cfg.scene_per_batch, 
                 k_shots = cfg.k_shots, 
                 seed= cfg.seed):
        """
        Construct N-shot dataset
        :param batch_size:  Experiment batch_size
        :param scene_per_batch: Integer indicating the number of scenes per batch
        :param k_shots: Integer indicating support samples per class
        :param seed: seed for random function
        :param shuffle: if shuffle the dataset
        """
        self.k_shots = k_shots
        self.scene_per_batch = scene_per_batch
        self.batch_size = batch_size 
        
        
        np.random.seed(seed)
        self.x = torch.load('data/X_'+str(cfg.scene_name)+'_scenarios_FMCW.pth')
        self.y = torch.load('data/Y_'+str(cfg.scene_name)+'_scenarios_FMCW.pth')        
        ratio_len = np.array([0.3,0.1,0.6]) * self.x.shape[0]

        self.x_train, self.x_val, self.x_test = self.x[:int(ratio_len[0])], self.x[int(ratio_len[0]):int(ratio_len[0])+int(ratio_len[1])], self.x[int(ratio_len[0])+int(ratio_len[1]):]
        self.y_train, self.y_val, self.y_test = self.y[:int(ratio_len[0])], self.y[int(ratio_len[0]):int(ratio_len[0])+int(ratio_len[1])], self.y[int(ratio_len[0])+int(ratio_len[1]):]        

        self.x_train = self.processes_batch(self.x_train, torch.mean(self.x_train), torch.std(self.x_train))
        self.x_test = self.processes_batch(self.x_test, torch.mean(self.x_test), torch.std(self.x_test))
        self.x_val = self.processes_batch(self.x_val, torch.mean(self.x_val), torch.std(self.x_val))

        self.batch_size = batch_size
        self.n_scenes = self.x.shape[0]
        self.n_classes = self.x.shape[1]
        self.shots_per_class = self.x.shape[2]
        
        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.datatset = {"train": self.x_train, "val": self.x_val, "test": self.x_test}
        self.datatset_y = {"train": self.y_train, "val": self.y_val, "test": self.y_test}

    def processes_batch(self, x_batch, mean, std):
        """
        Normalizes a batch images
        :param x_batch: a batch images
        :return: normalized images
        """
        return (x_batch - mean) / std

    def _sample_new_batch(self, data_pack, data_pack_y):
        """
        Collect 1000 batches data for N-shot learning
        :param data_pack: one of(train,test,val) dataset shape[n_scenes, n_classes, shots_per_class,512,60]
        :param data_pack_y: one of(train,test,val) dataset shape[n_scenes, n_classes, shots_per_class, 6]        
        :return: A list with [support_set_x,support_set_y,target_x,target_y] ready to be fed to our networks
        """
        support_set_x = np.zeros((self.batch_size, self.n_classes, self.k_shots, data_pack.shape[3],
                                  data_pack.shape[4]), np.float32)
        support_set_y = np.zeros((self.batch_size, self.n_classes, self.k_shots, self.n_classes), np.int32)
        
        query_x = np.zeros((self.batch_size, data_pack.shape[3], data_pack.shape[4]), np.float32)
        query_y = np.zeros((self.batch_size, self.n_classes), np.int32)

        for i in range(self.batch_size):
            scene_idx = np.arange(data_pack.shape[0])
            shots_idx = np.arange(data_pack.shape[1])
            class_idx = np.arange(self.n_classes)
            choose_scene = np.random.choice(scene_idx, size=1, replace=False)
            choose_class = np.random.choice(class_idx, size=1)
            choose_samples = np.random.choice(shots_idx, size=self.k_shots + 1, replace=False)

            x_temp = data_pack[choose_scene].squeeze(0)
            x_temp = x_temp[:, choose_samples] # n_classes * (k+1) * 512 * 60 
            y_temp = data_pack_y[choose_scene].squeeze(0)
            y_temp = y_temp[:, choose_samples] # n_classes * (k+1) * 6 
            
            # split support & query 
            support_set_x[i] = x_temp[:,:-1]
            support_set_y[i] = y_temp[:,:-1]
            
            query_x[i] = x_temp[choose_class,-1] # select classes and last shots 
            query_y[i] = y_temp[choose_class,-1] # select classes and last shots 

        # support_set_x: bs * n_classes * k * 512 * 60
        # support_set_y: bs * n_classes * k * 6
        # query_x: bs * 512 * 60 
        # query_y: bs * 6 
        return support_set_x, support_set_y, query_x, query_y


    def _get_batch(self, dataset_name):
        """
        Get next batch from the dataset with name.
        :param dataset_name: The name of dataset(one of "train","val","test")
        :param augment: if rotate the images
        :return: a batch images
        """
  
        support_set_x, support_set_y, query_x, query_y = self._sample_new_batch(self.datatset[dataset_name],
                                                                                self.datatset_y[dataset_name],)

        
        if cfg.reshape_to_scene == True:
            # support_set_x: bs * n_classes * k * 512 * 60 
            support_set_x = torch.from_numpy(support_set_x).permute(0,2,1,3,4).numpy()
            # support_set_x: (bs *  k) * n_classes * 512 * 60 
            support_set_x = support_set_x.reshape(support_set_x.shape[0]*support_set_x.shape[1],
                                                  support_set_x.shape[2],
                                                  support_set_x.shape[3],
                                                  support_set_x.shape[4])

            support_set_y = torch.from_numpy(support_set_y).permute(0,2,1,3).numpy()
            support_set_y = support_set_y.reshape(support_set_y.shape[0]*support_set_y.shape[1],
                                                  support_set_y.shape[2],
                                                  support_set_y.shape[3])
            support_set_y = np.array(torch.max(torch.Tensor(support_set_y), dim = 2)[1])

            query_y = np.array(torch.max(torch.Tensor(query_y),dim = 1)[1])            
            
        else:  
            support_set_x = support_set_x.reshape(support_set_x.shape[0],
                                                  support_set_x.shape[1]*support_set_x.shape[2],
                                                  support_set_x.shape[3],
                                                  support_set_x.shape[4])

            support_set_y = support_set_y.reshape(support_set_y.shape[0],
                                                  support_set_y.shape[1]*support_set_y.shape[2],
                                                  support_set_y.shape[3])
            support_set_y = np.array(torch.max(torch.Tensor(support_set_y), dim = 2)[1])

            query_y = np.array(torch.max(torch.Tensor(query_y),dim = 1)[1])

        
        return support_set_x, support_set_y, query_x, query_y

    def get_train_batch(self):
        return self._get_batch("train")

    def get_val_batch(self):
        return self._get_batch("val")

    def get_test_batch(self):
        return self._get_batch("test")

    


    