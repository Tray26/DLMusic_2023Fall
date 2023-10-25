import os
import json
import csv
import torch
import torchaudio
import argparse
import itertools
import time

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader

from model import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from meldataset import MelDataset, AttrDict
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
from Melspectrogram import mel_spectrogram






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

    with open('train_spec_config.json') as f:
        spec_config = f.read()

    train_config = json.loads(train_config)
    train_config = AttrDict(train_config)

    spec_config = json.loads(spec_config)
    spec_config = AttrDict(spec_config)

    torch.manual_seed(train_config.seed)
    torch.cuda.manual_seed(train_config.seed)

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device_name = 'mps' if torch.has_mps else 'cpu'
    print(device_name)
    device = torch.device(device_name)

    generator = Generator(train_config).to(device)
    discriminatorp = MultiPeriodDiscriminator().to(device)
    discriminators = MultiScaleDiscriminator().to(device)

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

    generator.train()
    discriminatorp.train()
    discriminators.train()

    for epoch in range(max(0, last_epoch), a.training_epochs):
        start = time.time()
        print("Epoch: {}".format(epoch+1))
        for wav_segment, mel in train_loader:
            wav_segment = torch.autograd.Variable(wav_segment.to(device, non_blocking=True))
            mel = torch.autograd.Variable(mel.to(device, non_blocking=True))

            print(wav_segment.shape, mel.shape)

            gen_wav = generator(mel)
            gen_wav_mel = mel_spectrogram(gen_wav, spec_config.n_fft, spec_config.num_mels,
                                          spec_config.sampling_rate, spec_config.hop_size, 
                                          spec_config.win_size, spec_config.fmin, spec_config.fmax)
            
            print(gen_wav.shape, gen_wav_mel.shape)
            
            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = discriminatorp(wav_segment, gen_wav.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = discriminators(wav_segment, gen_wav.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            print('ok')





    


    # if torch.cuda.is_available():
        
    #     # h.num_gpus = torch.cuda.device_count()
    #     train_config.batch_size = int(train_config.batch_size / train_config.num_gpus)
    #     print('Batch size per GPU :', train_config.batch_size)
    # else:
    #     pass
