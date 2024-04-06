from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# Additional Scripts
from utils import transforms as T
from utils.dataset import DentalDataset
from utils.utils import EpochCallback

from config import cfg

from train_transunet import TransUNetSeg

from ignite.metrics import ConfusionMatrix, IoU, DiceCoefficient

import pandas as pd
import numpy as np
from fuzzy_control import act_alpha
from focal_loss import FocalLoss

import os
from datetime import datetime

from torch.profiler import profile, record_function, ProfilerActivity
import time
from thop import profile



cm_IoU = ConfusionMatrix(num_classes=5)
cm_Dice = ConfusionMatrix(num_classes=5)
iou_metric = IoU(cm_IoU)
iou_no_bg_metric = iou_metric[1:]
mean_iou_no_bg_metric = iou_no_bg_metric.mean()
dice_metric = DiceCoefficient(cm_Dice)


DATA_DIR=os.getcwd()+"/"
MODEL_DIR = DATA_DIR +"pesos/"


class TrainTestPipe:
    def __init__(self, train_path, test_path, model_path, device):
        self.device = device
        self.model_path = "model_path"

        self.train_loader = self.__load_dataset(train_path, train=True)
        self.test_loader = self.__load_dataset(test_path)
        self.log = {}
        self.hist = pd.DataFrame()

        self.transunet = TransUNetSeg(self.device)
        
        self.best_val_loss = float('inf')  # Initialize with a very large value

    def __load_dataset(self, path, train=False):
        shuffle = False
        transform = False

        if train:
            shuffle = True
            #transform = transforms.Compose([T.RandomAugmentation(2)])

        set = DentalDataset(path, transform)
        loader = DataLoader(set, batch_size=cfg.batch_size, shuffle=shuffle)

        return loader

    def __loop(self, loader, step_func, t):
        total_loss = 0

        
        
        for step, data in enumerate(loader):
            img, mask = data['img'], data['mask']
            img = img.to(self.device)
            mask = mask.to(self.device)

            loss, cls_pred = step_func(img=img, mask=mask, iou_metric=iou_metric, dice_metric=dice_metric)

            total_loss += loss

            t.update()

        return total_loss
    

    def train(self):
        delta_arreglo=[]
        start_alpha = np.array([0.5,0.5,0.5,0.5,0.5])#0.1,0.2,1.4,1.4,1.25
        alpha = start_alpha
        flag_alpha = True
        class_sizes = np.array([0.95341027, 0.03366332, 0.00378282, 0.00374522, 0.00539836])
        callback = EpochCallback(self.model_path, cfg.epoch,
                                 self.transunet.model, self.transunet.optimizer, 'test_loss', cfg.patience)

        #aqui poner el print para ver parametros entrenables

        # Cuenta los parámetros entrenables
        trainable_params = sum(p.numel() for p in self.transunet.model.parameters() if p.requires_grad)

        print(f'Número de parámetros entrenables: {trainable_params}')
        print("##############################################")
        #dd = self.transunet.model
        #inp = torch.rand(1,3,256,256).to('cuda:0')
        #flops, params = profile(dd, inputs=(inp, ))
        #print (f'Número de FLOPS: {flops}')
        
        current_datetime = datetime.now()
        timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M-%S.%f")
        
        
        for epoch in range(cfg.epoch):
            
            with tqdm(total=len(self.train_loader) + len(self.test_loader)) as t:
                
                #start_time = time.time()  # Start time measurement
                train_loss = self.__loop(self.train_loader, self.transunet.train_step, t)
                #forward_time = time.time() - start_time  # Time taken for forward pass
                #print("Eppoch time:", forward_time)
                
                IoU = iou_no_bg_metric.compute().numpy()
                mIoU = mean_iou_no_bg_metric.compute().numpy()
                Dice = dice_metric.compute().numpy()
                
                iou_metric.reset()
                dice_metric.reset()
                
                test_loss = self.__loop(self.test_loader, self.transunet.test_step, t)
                
                val_IoU = iou_no_bg_metric.compute().numpy()
                val_Dice = dice_metric.compute().numpy()
                val_mIoU = mean_iou_no_bg_metric.compute().numpy()

            self.log['loss'] = train_loss / len(self.train_loader)
            self.log['IoU'] = IoU
            self.log['mIoU'] = mIoU
            self.log['Dice'] = Dice
            self.log['alpha'] = alpha
            self.log['val_loss'] = test_loss / len(self.test_loader)
            self.log['val_IoU'] = val_IoU
            self.log['val_mIoU'] = val_mIoU
            self.log['val_Dice'] = val_Dice
            
            self.hist = self.hist._append(self.log, ignore_index = True)
            self.hist.index.name = "epoch"
            
            callback.epoch_end(epoch + 1,
                               {'loss': train_loss / len(self.train_loader),
                                'test_loss': test_loss / len(self.test_loader),
                                'IoU': IoU,
                                'mIoU': mIoU,
                                'Dice': Dice,
                                'val_IoU': val_IoU,
                                'val_mIoU': val_mIoU,
                                'val_Dice': val_Dice,
                                'alpha': alpha,
                                },
                               self.hist)
            
            if callback.end_training:
                break
            
            if epoch == 0:
                continue
            
            delta_arreglo.append((self.hist['loss'].iloc[-1]-self.hist['loss'].iloc[-2]))
            
            delta_alpha = [act_alpha(class_sizes[i],self.hist['loss'].iloc[-2],self.hist['loss'].iloc[-1]) for i in range(len(class_sizes))]
            alpha = start_alpha + delta_alpha #alpha += delta_alpha ##sustituimos el alpha directamente
            
            
            self.transunet.criterion = FocalLoss(gamma=2, alpha=alpha.tolist())
            
            # Compare the validation loss with the best validation loss so far
            if self.log['val_loss'] < self.best_val_loss:
               self.best_val_loss = self.log['val_loss']
               # Save the model
               #torch.save(self.transunet.model.state_dict(), MODEL_DIR + "_" +timestamp+"_" + str(epoch)+"_" +'best_model.pth')
               torch.save({'model_state_dict': self.transunet.model.state_dict(), 
                           'optimizer_state_dict': self.transunet.optimizer.state_dict()},
                          MODEL_DIR + "_" +timestamp+"_" + str(epoch)+"_" +'best_model.pth')
              # torch.save(model.state_dict(), MODEL_DIR + "model_" + str(opt) + "_" + str(lr) + "_" + str(epoch) + ".pth")
            
            
        delta_arreglo.sort()
        print("Valor mas chico de delta_alpha: ",delta_arreglo[0],"\n")
        print("Valor mas grande de delta_alpha: ",delta_arreglo[-1],"\n")

            
