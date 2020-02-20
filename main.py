import argparse
from train import Trainer
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper parameters setting')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--way', type=int, default=20)
    parser.add_argument('--times', type=int, default=200)
    args = parser.parse_args()
    print(vars(args))

    cuda_device = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

    trainer = Trainer(args)
    trainer.train()
