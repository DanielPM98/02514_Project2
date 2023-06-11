import torch
import argparse

from train import Trainer
from evaluate import Evaluator

def get_args():
    parser = argparse.ArgumentParser(description='Model args')    

    #Mode
    parser.set_defaults(train=False)
    parser.set_defaults(evaluate=False)
    parser.add_argument('--train',
                        dest='train',
                        action='store_true')
    parser.add_argument('--eval',
                        dest='eval',
                        action='store_true')
    
    # Data
    parser.add_argument('--weights_path',
                        type=str,
                        help='path to the model weights',
                        default=None)
    parser.add_argument('--save_dir',
                        type=str,
                        help='path to save the results',
                        default='./logs/')
    parser.add_argument('--resolution',
                        type=int,
                        help='Resolution of the images for training',
                        default=256)
    parser.add_argument('--dataset_path',
                        type=str,
                        help='path to root dataset',
                        )
    parser.add_argument('--dataset',
                        type=str,
                        help='Name of the dataset',
                        choices=['ph2', 'drive'])
    # parser.add_argument('--augmentation',
    #                     type=bool,
    #                     help='Load dataset with augmented data',
    #                     default=True)
    parser.add_argument('--name',
                        type=str,
                        help='name of the model',
                        default='default')
    
    # Model
    parser.add_argument('--model',
                        type=str,
                        help='Choose a model from modules',
                        default='baseline')
    # Model params
    # parser.add_argument('--dropout',
    #                     type=float,
    #                     help='dropout regularization value',
    #                     default=0.0)
    # parser.add_argument('--freeze',
    #                     type=bool,
    #                     help='Freeze initial layers',
    #                     default=True)
    
    # Optimization
    parser.add_argument('--batch_size',
                        type=int,
                        help='batch size of for the training',
                        default=8)
    parser.add_argument('--learning_rate',
                        type=float,
                        help='learning rate',
                        default=1e-4)
    parser.add_argument('--num_epochs',
                        type=int,
                        help='number of epochs',
                        default=50)
    parser.add_argument('--scheduler_step_size',
                        type=int,
                        help='step size of the scheduler',
                        default=7)
    parser.add_argument('--optimizer',
                        type=str,
                        choices=['sgd', 'adam'],
                        default='adam')
    
    #System
    # parser.add_argument('--num_workers',
    #                     type=int,
    #                     help='number of dataloader workers',
    #                     default=1)
    parser.add_argument('--device',
                        type=str,
                        help='Device to run the model',
                        default='cuda:0')
    
    return parser.parse_args()


def main():
    args = get_args()
    print(args)

    if args.train:
        model_trainer = Trainer(args)
        model_trainer.train()
    elif args.eval:
        print('Eval Mode')
        model_eval = Evaluator(args)
        model_eval.eval()
    print('===== DONE =====')


if __name__ == '__main__':
    main()
    