import argparse

from simclr.models.pytorch.resnet import ResNet

import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    args = parser.parse_args()

    model = ResNet()

    temp_img =  torch.zeros(1,3,224,224)
    out = model.forward(temp_img)
    print(out.shape)

if __name__ == '__main__':
    main()