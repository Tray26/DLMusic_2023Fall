import os
import csv
import numpy as np
import torch
from testloader import get_testloader
from model_shortChunkCNN import ShortChunkCNN


if __name__ == "__main__":
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(device_name)
    net = ShortChunkCNN().to(device)
    test_loader = get_testloader()
    best_model = 'best_model_short_BCE_400epochs.ckpt'

    S = torch.load(os.path.join('./models', best_model))
    net.load_state_dict(S)
    print('loaded')

