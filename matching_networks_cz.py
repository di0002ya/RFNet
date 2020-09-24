import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import config as cfg
from torch.nn.parallel import DistributedDataParallel
    # for Single-Process Multi-GPU case
# torch.distributed.init_process_group(backend="nccl") #initialize torch.distributed

    
class AttentionalClassify(nn.Module):
    def __init__(self):
        super(AttentionalClassify, self).__init__()

    def forward(self, similarities, support_set_y):
        """
        Products pdfs over the support set classes for the target set image.
        :param similarities: A tensor with cosine similarites of size[batch_size,sequence_length]
        :param support_set_y:[batch_size,sequence_length,classes_num]
        :return: Softmax pdf shape[batch_size,classes_num]
        """
#         softmax = nn.Softmax()
#         softmax_similarities = softmax(similarities)
        preds = similarities.unsqueeze(1).bmm(support_set_y).squeeze()
        return preds


class DistanceNetwork(nn.Module):
    """
    This model calculates the cosine distance between each of the support set embeddings and the target image embeddings.
    """

    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image):
        """
        forward implement
        :param support_set:the embeddings of the support set images.shape[sequence_length,batch_size,64]
        :param input_image: the embedding of the target image,shape[batch_size,64]
        :return:shape[batch_size,sequence_length]
        """
        eps = 1e-10
        similarities = []
        for support_image in support_set:
            sum_support = torch.sum(torch.pow(support_image, 2), 1)
            support_manitude = sum_support.clamp(eps, float("inf")).rsqrt()
            dot_product = input_image.unsqueeze(1).bmm(support_image.unsqueeze(2)).squeeze()
            cosine_similarity = dot_product * support_manitude
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        return similarities.t()


class MatchingNetwork(nn.Module):
    def __init__(self, base_model,fce,use_cuda):
        """
        This is our main network
        :param keep_prob: dropout rate
        :param batch_size:
        :param num_channels:
        :param learning_rate:
        :param fce: Flag indicating whether to use full context embeddings(i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set:
        :param num_samples_per_class:
        :param image_size:
        """
        super(MatchingNetwork, self).__init__()
        self.batch_size = cfg.batch_size        
        self.g = base_model.Net(cfg.k_shots, cfg.num_class, cfg.win_len)
        self.dn = DistanceNetwork()
        self.fce = fce
        self.classify = AttentionalClassify()
        if self.fce:
            # self.lstm = DistributedDataParallel(nn.LSTM(cfg.hid_dim, cfg.hid_dim ,32, bidirectional=True).cuda())
            # self.lstm = nn.DataParallel(nn.LSTM(cfg.hid_dim, cfg.hid_dim ,32, bidirectional=True))
            self.lstm = nn.LSTM(cfg.hid_dim, cfg.hid_dim ,32, bidirectional=True)
            
        self.fc = nn.Linear(cfg.num_class,cfg.num_class)
        # self.sim_net = nn.Linear(3,6)
        self.sim_net = nn.Linear(3,12)
    def forward(self, support_set_images, support_set_y_one_hot, target_image, target_y):
        """
        Main process of the network
        :param support_set_images: shape[batch_size,sequence_length,num_channels,image_size,image_size]
        :param support_set_y_one_hot: shape[batch_size,sequence_length,num_classes_per_set]
        :param target_image: shape[batch_size,num_channels,image_size,image_size]
        :param target_y:
        :return: 
        """
        # produce embeddings for support set images
        encoded_time = []
        encoded_freq = []        
        encoded_fuse = []        
        
        
        for i in np.arange(support_set_images.size(1)):
            _ , gen_encode = self.g(support_set_images[:, i, :, :])

            feat_time, feat_freq, feat = gen_encode
            encoded_time.append(feat_time)
            encoded_freq.append(feat_freq)            
            encoded_fuse.append(feat)
            
            
        # produce embeddings for target images
        preds_direct , gen_encode = self.g(target_image)
        feat_time, feat_freq, feat = gen_encode
        encoded_time.append(feat_time)
        encoded_freq.append(feat_freq)            
        encoded_fuse.append(feat)
            

        output_time = torch.stack(encoded_time)
        output_freq = torch.stack(encoded_freq)        
        output_fuse = torch.stack(encoded_fuse)        
        
        del encoded_time, encoded_freq, encoded_fuse, gen_encode, support_set_images, target_image
        torch.cuda.empty_cache()
                
                
                
    
        # use fce? 
        if self.fce:
            #self.lstm.module.flatten_parameters()                      
            output_fuse, _ = self.lstm(output_fuse) # for fast computations
            
            
        # get similarities between support set embeddings and target
        similarites_fuse = self.dn(support_set=output_fuse[:-1], input_image=output_fuse[-1])
        similarites_time = self.dn(support_set=output_time[:-1], input_image=output_time[-1])
        similarites_freq = self.dn(support_set=output_freq[:-1], input_image=output_freq[-1])
        
        del output_fuse, output_time, output_freq
        torch.cuda.empty_cache()     
        
        # get overall similarities
        # [batch_size,classes_num]
#         # Trial 1: sim_net 18 -> 6 
#         combine = torch.cat([similarites_fuse, similarites_time, similarites_freq],-1)
#         similarites = self.sim_net(combine)

#         # Trial 2: sim_net: 3 -> 6
        combine = torch.cat([similarites_fuse.unsqueeze(-1), similarites_time.unsqueeze(-1), similarites_freq.unsqueeze(-1)],-1) # batch_size, class_num, 3
        similarites = self.sim_net(combine).sum(-1) 
        
        
        
    
        # produce predictions for target probabilities
        preds = self.classify(similarites, support_set_y=support_set_y_one_hot)
        
        del support_set_y_one_hot        
        torch.cuda.empty_cache()        
        
        preds = self.fc(preds) + preds_direct
        


        # calculate the accuracy
        indices = preds.max(1)[1]
        accuracy = torch.mean((indices.squeeze() == target_y).float())
        crossentropy_loss = F.cross_entropy(preds, target_y.long())

        return accuracy, crossentropy_loss

