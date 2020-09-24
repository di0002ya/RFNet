# data = 'UWB'
# data = 'FMCW'
data = 'WIFI'
batch_size = 'Please set'
scene_per_batch = 'Please set'
k_shots = 'Please set'
seed = 2017 
scene_name = "100"

act = 'relu'
base_net = 'fuse_sep_att_cz'
build_net = 'matching_cz'

############
# Matching Network
############
fce = True
reshape_to_scene = True
matching_net = '3'    

    
update_step = 2
update_lr = 1e-3
meta_lr = 1e-4

num_class = 6
use_para = 'Please set'
hid_dim = 512
win_len = 512


cnn = 'resnet18'

total_train_batches = 'Please set'
total_val_batches = 'Please set'
total_test_batches = 'Please set'
total_epochs = 'Please set'

####################
# Optimizer Tuning
####################
lr = 0.001 # 0.005 #0.001
b1 = 0.5
b2 = 0.999
optim = 'adam'

