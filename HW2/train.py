import os
import json
import csv
import torch
import torchaudio
import argparse
import itertools

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader

from model import Generator, DiscriminatorP, DiscriminatorS
from meldataset import MelDataset, AttrDict
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint







if __name__ == '__main__':
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()


    with open('train_config.json') as f:
        train_config = f.read()

    train_config = json.loads(train_config)
    train_config = AttrDict(train_config)

    torch.manual_seed(train_config.seed)
    torch.cuda.manual_seed(train_config.seed)

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    generator = Generator(train_config).to(device)
    discriminatorp = DiscriminatorP().to(device)
    discriminators = DiscriminatorS().to(device)

    os.makedirs(a.checkpoint_path, exist_ok=True)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')
    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        discriminatorp.load_state_dict(state_dict_do['discriminatorp'])
        discriminators.load_state_dict(state_dict_do['discriminators'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    optim_g = torch.optim.AdamW(generator.parameters(), train_config.learning_rate, betas=[train_config.adam_b1, train_config.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(discriminators.parameters(), discriminatorp.parameters()),
                                train_config.learning_rate, betas=[train_config.adam_b1, train_config.adam_b2])
    
    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=train_config.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=train_config.lr_decay, last_epoch=last_epoch)

    train_dataset = MelDataset(
        data_path='./m4singer', sampling_period=5, real_batch_size=train_config.batch_size
    )

    train_loader = DataLoader(
        train_dataset, num_workers=train_config.num_workers,
        shuffle=True, batch_size=train_config.batch_size, drop_last=True
    )

    valid_dataset = MelDataset(
        data_path='./m4singer_valid', sampling_period=5, real_batch_size=train_config.batch_size, split='valid'
    )

    valid_loader = DataLoader(
        valid_dataset, num_workers=train_config.num_workers,
        shuffle=False, batch_size=train_config.batch_size, drop_last=True
    )

    sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    


    # if torch.cuda.is_available():
        
    #     # h.num_gpus = torch.cuda.device_count()
    #     train_config.batch_size = int(train_config.batch_size / train_config.num_gpus)
    #     print('Batch size per GPU :', train_config.batch_size)
    # else:
    #     pass