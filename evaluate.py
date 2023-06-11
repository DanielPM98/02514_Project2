import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks
import pandas as pd


import os
import time

from data.dataset import get_dataloaders
from modules import baseline, metrics, unet


class Evaluator:
    def __init__(self, args, seed=42) :

        self.seed = seed

        self.create_logs(args)

        # Set device to run model
        if (args.device == 'cpu') or not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(args.device)

        try:
            self.model = torch.load(args.weights_path)
        except Exception as e:
            print('Weights path is not valid. Required model weights')

        # Model params
        params = {
            'channels': 3,
            'out_channels': 1,
                  }

        # Load model
        self.model = torch.load(args.weights_path)
        self.model.to(self.device) 

        _, _, self.test_loader = get_dataloaders(
            root=args.dataset_path,
            name=args.dataset,
            resolution=args.resolution,
            batch_size=args.batch_size,
            seed=self.seed
        )

        self.result_metrics = metrics.Metrics()


    def eval(self):
        print('===== Comencing evaluation =====')
        torch.cuda.empty_cache()
        self.model.eval()
   
        with torch.no_grad():
            for idx, data in enumerate(self.test_loader):
                image, target = self._unpack_and_move(data)

                prediction = self.model(image)
                
                # prediction, target = self._parser_data(prediction, target)
                self.result_metrics.add(target, prediction)

                self._save_batch_prediction(idx, image, target, prediction)

 
        #Report 
        current_time = time.strftime('%H:%M', time.localtime())
        print(f'[{current_time}]')
        self._save_metrics()
        

    def _save_batch_prediction(self, idx, img_list, target, pred):
        samples = len(img_list)
        color = [0, 1, 0]
        plt.figure(figsize=(10,4))
        for i in range(samples):
            unint_img = torch.clamp(torch.round(img_list[i] * 255), min=0, max=255).to(torch.uint8)
            mask = target[i].gt(0).to(torch.bool)
            overlayed_img = draw_segmentation_masks(unint_img.cpu(), 
                                                    mask.cpu(), 
                                                    alpha=0.6, 
                                                    colors=(0,255,0))
            plt.subplot(2, 8, i+1)
            plt.imshow(overlayed_img.permute(1,2,0))
            plt.axis('off')
            plt.title(f'Original Mask')

            mask = pred[i].gt(0).to(torch.bool)
            overlayed_img = draw_segmentation_masks(unint_img.cpu(), 
                                                    mask.cpu(), 
                                                    alpha=0.6, 
                                                    colors=(0,255,0))
            plt.subplot(2, 8, i+9)
            plt.imshow(overlayed_img.permute(1,2,0))
            plt.axis('off')
            plt.title(f'Prediction')

        plt.savefig(os.path.join(self.save_dir, f'batch_pred_{idx}.png'))

    def _save_batch_overlay(self, idx, img_list, target, pred):
        samples = len(img_list)
        colors = [[0, 255, 0], [255, 0, 0]]
        plt.figure(figsize=(10,2))
        for i in range(samples):
            unint_img = torch.clamp(torch.round(img_list[i] * 255), min=0, max=255).to(torch.uint8)
            target_mask = target[i].gt(0).to(torch.bool)
            pred_mask = pred[i].gt(0).to(torch.bool)
            mask = torch.cat((target_mask, pred_mask), dim=0)

            overlayed_img = draw_segmentation_masks(unint_img.cpu(), 
                                                    mask.cpu(), 
                                                    alpha=0.3, 
                                                    colors=colors)
            plt.subplot(1, 8, i+1)
            plt.imshow(overlayed_img.permute(1,2,0))
            plt.axis('off')
            plt.title(f'Segmentation')

        plt.savefig(os.path.join(self.save_dir, f'overlay_batch_pred_{idx}.png'))


    def _save_metrics(self):
        average_metrics = self.result_metrics.mean()
        # Print metrics on the terminal
        print(f'Accuracy: {average_metrics["accuracy"]}')
        print(f'Dice score: {average_metrics["dice_score"]}')
        print(f'Specifity: {average_metrics["specifity"]}')
        print(f'Sensitivity: {average_metrics["sensitivity"]}')
        print(f'IoU: {average_metrics["iou"]}')

        for key, value in average_metrics.items():
            average_metrics[key] = value.cpu().item()

        df = pd.DataFrame([average_metrics])
        df.to_csv(os.path.join(self.save_dir, 'metrics.txt'))
        print('All results have been saved.')

    

    def _parser_data(self, prediction, target):
        prediction = prediction
        target = (target > 0).float()
        return prediction, target
    
    
    def _unpack_and_move(self, data):
        image = data['image'].to(self.device)
        target = data['label'].to(self.device)

        return image, target
    
    def create_logs(self, args):
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)
        self.save_dir = os.path.join(args.save_dir, args.name)

        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.save_dir = os.path.join(self.save_dir, 'val')
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        print("Save files' path: ", self.save_dir)
        