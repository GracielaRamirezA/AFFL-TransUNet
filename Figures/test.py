# This is a comment
from easydict import EasyDict

cfg = EasyDict()
cfg.batch_size = 8 #originales  16
cfg.epoch = 1200  #200
cfg.learning_rate = 1e-6
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.patience = 50
cfg.inference_threshold = 0.75

cfg.transunet = EasyDict()
cfg.transunet.img_dim = 256
cfg.transunet.in_channels = 3
cfg.transunet.out_channels = 128
cfg.transunet.head_num = 2
cfg.transunet.mlp_dim = 64
cfg.transunet.block_num = 1
cfg.transunet.patch_dim = 16
cfg.transunet.class_num = 5
