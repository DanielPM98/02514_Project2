import torch
from tqdm import tqdm
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import torch.nn.functional as F

import time
import copy
import os

from modules import baseline, metrics, losses, unet
from data.dataset import get_dataloaders


class Trainer:
    def __init__(self, args, seed=42):

        # Creates save directory and logs
        self.create_logs(args)

        self.max_epochs = args.num_epochs
        self.seed = seed

        # Set device to run model
        if (args.device == 'cpu') or not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(args.device)
        
        # Model params
        params = {
            'channels': 3,
            'out_channels': 1,
                  }

        # Select model
        if args.model == 'baseline':
            self.model = baseline.SegNetS(params)
        elif args.model == 'unet':
            self.model = unet.UNet(params['channels'], 1)

        self.model.to(self.device) 

        self.train_loader, self.val_loader, _ = get_dataloaders(
            root=args.dataset_path,
            name=args.dataset,
            resolution=args.resolution,
            batch_size=args.batch_size,
            seed=self.seed
        )

        if args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             args.learning_rate,
                                             momentum=0.9)
        elif args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                            args.learning_rate,
                                            weight_decay=1e-3)
        else:
            raise AssertionError('Inserted optimizer is not valid')
        
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            args.scheduler_step_size,
                                                            gamma=0.1)
        

        self.loss_fn =  nn.BCEWithLogitsLoss(weight=torch.Tensor([2.0]).to(self.device))
        self.best_loss = 100000
        self.best_weights = None # TODO: CHANGE LATER TO LOAD MODEL
        
        self.result_dict = {'train_metric': [],
              'val_metric': [],
              'train_loss': [],
              'test_loss': [],
               'mean_time': 0,
               'resolution': args.resolution,
               'batch_size': args.batch_size,
               'lr': args.learning_rate}
        
    
    def train(self):
        torch.cuda.empty_cache()
        self.start_time = time.time()
        for epoch in range(self.max_epochs):
            current_time = time.strftime('%H:%M', time.localtime())
            print('{} - Epoch {}'.format(current_time, epoch))

            avg_loss, avg_metric = self.batch_loop(self.train_loader, train=True)
            self._save_loss_metric(avg_loss, avg_metric, mode='train')

            current_time = time.strftime('%H:%M', time.localtime())
            print(f'{current_time} - Training Loss: {self.result_dict["train_loss"][-1]:3.4f}')
            
            self.lr_scheduler.step()

            if self.val_loader is not None:
                avg_loss, avg_metric = self.batch_loop(self.val_loader, train=True)
                self._save_loss_metric(avg_loss, avg_metric, mode='val')

                current_time = time.strftime('%H:%M', time.localtime())
                print(f'{current_time} - Validation Loss: {self.result_dict["val_loss"][-1]:3.4f}')

        total_time = time.time() - self.start_time
        self.result_dict['mean_time'] = total_time / self.max_epochs
        self.save_model()
        self.save_results()


    def batch_loop(self, dataloader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

        accumulated_loss = 0.0
        accumulated_metric = metrics.Metrics()

        for _, data in enumerate(dataloader):
            image, target = self._unpack_and_move(data)
            
            if train:
                self.optimizer.zero_grad()

            prediction = self.model(image)
            
            # Parser data
            # target, prediction = self._parser_data(target, prediction)

            # self._debug_saves(image.cpu(), target.cpu(), prediction.cpu())

            loss_value = self.loss_fn(prediction, target)

            if train:
                loss_value.backward()
                self.optimizer.step()

            accumulated_loss += loss_value.item()
            accumulated_metric.add(target, prediction)

        #Report 
        average_loss = accumulated_loss / (len(dataloader.dataset) )
        average_metric = accumulated_metric.mean()

        return average_loss, average_metric
    

    def _save_loss_metric(self, loss, metric, mode):
        self.result_dict[f'{mode}_loss'].append(loss)
        self.result_dict[f'{mode}_metric'].append(metric)

        # for key, value in metric.items():
        #     self.result_dict[f'{mode}_metric'][key].append(value.cpu().item()) 


    def _parser_data(self, target, pred):
        pred = pred
        target = (target > 0).float()
        return target, pred
    

    def _unpack_and_move(self, data):
        image = data['image'].to(self.device)
        target = data['label'].to(self.device)

        return image, target
    
    
    def save_model(self):
        best_model = os.path.join(self.save_dir,
                                      'weights.pth')
        self.model.load_state_dict(self.best_weights)
        torch.save(self.model, best_model)
        print('Model saved.')


    def save_results(self):
        train_metrics = self.result_dict['train_metric']
        val_metrics = self.result_dict['val_metric']
        del self.result_dict['train_metric'], self.result_dict['val_metric']

        df = pd.DataFrame(self.result_dict)
        df.to_csv(os.path.join(self.save_dir, 'results.txt'))

        df = pd.DataFrame(train_metrics)
        df.to_csv(os.path.join(self.save_dir, 'train_metrics.txt'))

        df = pd.DataFrame(val_metrics)
        df.to_csv(os.path.join(self.save_dir, 'val_metrics.txt'))

        fig = plt.figure(figsize=(10,10))
        plt.title('Losses')
        plt.plot(self.result_dict['train_loss'], label='train loss')
        plt.plot(self.result_dict['test_loss'], label='val loss')

        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'results.png'))
        
    
    def create_logs(self, args):
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)
        self.save_dir = os.path.join(args.save_dir, args.name)

        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
            os.mkdir(os.path.join(self.save_dir, 'train'))
            print("Save files' path: ", os.path.join(self.save_dir, 'train'))

        self.save_dir = os.path.join(self.save_dir, 'train')


    def _debug_saves(self, img, target, pred):
        fig = plt.figure(figsize=(14, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(img[0].permute(1,2,0))
        plt.title('Original Image')
        plt.subplot(1, 3, 2)
        plt.imshow(target[0].permute(1,2,0), cmap='gray')
        plt.title('Target mask')
        plt.subplot(1, 3, 3)
        plt.imshow(pred[0].permute(1,2,0).detach().numpy(), cmap='gray')
        plt.title('Prediction')

        fig.savefig(os.path.join(self.save_dir, 'debug.png'))
    
