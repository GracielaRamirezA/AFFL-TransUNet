import torch
from torch.optim import SGD, Adam

# Additional Scripts
from utils.transunet import TransUNet
from utils.utils import dice_loss
from config import cfg

from focal_loss import FocalLoss
import time

class TransUNetSeg:
    def __init__(self, device):
        self.device = device
        self.model = TransUNet(img_dim=cfg.transunet.img_dim,
                               in_channels=cfg.transunet.in_channels,
                               out_channels=cfg.transunet.out_channels,
                               head_num=cfg.transunet.head_num,
                               mlp_dim=cfg.transunet.mlp_dim,
                               block_num=cfg.transunet.block_num,
                               patch_dim=cfg.transunet.patch_dim,
                               class_num=cfg.transunet.class_num).to(self.device)

        self.criterion = FocalLoss(gamma=2, alpha=[0.5,0.5,0.5,0.5,0.5])#dice_loss 0.5,0.5,0.5,0.5,0.5
        self.optimizer = Adam(self.model.parameters(), lr=cfg.learning_rate)
        #self.optimizer = SGD(self.model.parameters(), lr=cfg.learning_rate,
                             #momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    def load_model(self, path):
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        self.model.eval()

    def train_step(self, **params):
        self.model.train()
        
        #start_time = time.time()  # Start time measurement

        self.optimizer.zero_grad()        
        pred_mask = self.model(params['img'])        
        params['iou_metric'].update((pred_mask, torch.argmax(params['mask'], dim=1)))
        params['dice_metric'].update((pred_mask, torch.argmax(params['mask'], dim=1)))        
        
        #forward_time = time.time() - start_time  # Time taken for forward pass
        #print("Forward pass time:", forward_time)
    
        loss = self.criterion(pred_mask, torch.argmax(params['mask'], dim=1))        
        loss.backward()
        
        #backward_time = time.time() - (start_time + forward_time)  # Time taken for backward pass
        #print("Backward pass time:", backward_time)
        
        self.optimizer.step()
        
        #optimizer_time = time.time() - (start_time + forward_time + backward_time)  # Time taken for optimizer step
        #rint("Optimizer step time:", optimizer_time)

        #total_time = time.time() - start_time  # Total time taken for the function execution
        #print("Total time:", total_time)
        

        return loss.item(), pred_mask

    def test_step(self, **params):
        self.model.eval()

        pred_mask = self.model(params['img'])
        params['iou_metric'].update((pred_mask, torch.argmax(params['mask'], dim=1)))
        params['dice_metric'].update((pred_mask, torch.argmax(params['mask'], dim=1)))
        
        loss = self.criterion(pred_mask, torch.argmax(params['mask'], dim=1))

        return loss.item(), pred_mask
