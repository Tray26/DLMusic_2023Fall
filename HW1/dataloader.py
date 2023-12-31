# import matplotlib.pyplot as plt
import os
import random
import torch
import numpy as np
import soundfile as sf
from torch.utils import data
import csv
from torchaudio_augmentations import (
    RandomResizedCrop,
    RandomApply,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
    Compose,
)




class SingerDataset(data.Dataset):
    # def __init__(self,data_path, split, num_samples, num_chunks, is_augmentation, 
    # support_data_path = './split_data'):
    # Need: sampling rate, sample time interval, augmentation, support_data_path
    def __init__(self,data_path, split, num_chunks, is_augmentation, sample_interval, 
                 sample_rate=16000, support_data_path = './support_data'):
        self.data_path =  data_path if data_path else ''        # data path
        self.split = split                                      # train or valid
        self.num_samples = int(sample_rate * sample_interval)   # total sampling data point
        self.sample_rate = sample_rate
        self.sample_interval = sample_interval
        self.num_chunks = num_chunks
        self.is_augmentation = is_augmentation                  #
        with open(os.path.join(support_data_path, 'singers.csv'), newline='') as readsingers:
            singers_list = csv.reader(readsingers)
            singers_list = list(singers_list)
            singers_list = singers_list[0]
        self.singers = singers_list                                 #singers list
        self.support_data_path = support_data_path
        self._get_song_info()
        if is_augmentation:
            self._get_augmentations()

    def _get_augmentations(self):                             # data augmentation
        transforms = [
            RandomResizedCrop(n_samples=self.num_samples),
            RandomApply([PolarityInversion()], p=0.8),
            RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3),
            RandomApply([Gain()], p=0.2),
            RandomApply([HighLowPass(sample_rate=self.sample_rate)], p=0.8),
            RandomApply([Delay(sample_rate=self.sample_rate)], p=0.5),
            RandomApply([PitchShift(n_samples=self.num_samples, sample_rate=self.sample_rate)], p=0.4),
            RandomApply([Reverb(sample_rate=self.sample_rate)], p=0.3),
        ]
        self.augmentation = Compose(transforms=transforms)

    # def _get_song_info
    def _get_song_info(self):
        list_filename = os.path.join(self.support_data_path, '%s.txt' % self.split)
        with open(list_filename) as f:
            lines = f.readlines()
        self.song_list = [line.strip() for line in lines]

    def _adjust_audio_length(self, wav):
        if self.split == 'train':
            # print(len(wav))
            # random_index = random.randint(0, len(wav) - self.num_samples - 1)
            random_index = int(np.floor(np.random.random(1) * (len(wav)-self.num_samples)))
            wav = wav[random_index : random_index + self.num_samples]
        else:
            chunk_number = len(wav) // self.num_samples
            hop = (len(wav) - self.num_samples) // chunk_number
            wav = np.array([wav[i * hop : i * hop + self.num_samples] for i in range(chunk_number)])
            wav = wav[0,:]
            # print(wav.shape)
        return wav

    def __getitem__(self, index):
        line = self.song_list[index]

        # get singer
        singer_name = line.split(', ')[1]
        # singer_name = singer_name[:-1]
        singer_index = self.singers.index(singer_name)

        # get audio
        audio_filename = line.split(', ')[0]
        # audio_filename = audio_filename[:-1]
        wav, fs = sf.read(audio_filename)
        # print(type(wav), wav.shape)
        if len(wav) < self.num_samples:
            # print('Audio not long enough:', line.split(', ')[2])
            len_mul = self.num_samples // len(wav)
            wav = np.repeat(wav, len_mul + 1)


        # wav, fs = torchaudio.load(audio_filename)

        # adjust audio length
        song_name = line.split(', ')[2]
        # print(song_name)
        # print(wav.shape, singer_index)
        wav = self._adjust_audio_length(wav).astype('float32')

        # plt.figure()
        # plt.plot(wav)
        # plt.show()

        # print(wav.shape, singer_index)

        # data augmentation
        if self.is_augmentation:
            wav = self.augmentation(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()

        
        # print(singer_index, singer_name, song_name)
        print(type(singer_index))
        return wav, singer_index

    def __len__(self):
        return len(self.song_list)

def get_dataloader(data_path='./artist20/mp3s-32k/',
                   split='train',
                   num_chunks=1,
                   batch_size=16,
                   num_workers=4,
                   is_augmentation=False,
                   sample_interval = 3.69):
    is_shuffle = True if (split == 'train') else False
    # num_chunks = int(sample_interval * 16000) 
    batch_size = batch_size if (split == 'train') else (batch_size // num_chunks)
    # print(split, batch_size)
    data_loader = data.DataLoader(dataset=SingerDataset(data_path, 
                                                        split, 
                                                        num_chunks, 
                                                        is_augmentation, 
                                                        sample_interval,
                                                       ),
                                  batch_size=batch_size,
                                  shuffle=is_shuffle,
                                  drop_last=True,
                                  num_workers=num_workers)
    return data_loader


if __name__ == "__main__":
    train_loader = get_dataloader(split='train', is_augmentation=False)
    iter_train_loader = iter(train_loader)
    train_wav, train_singer = next(iter_train_loader)

    # valid_loader = get_dataloader(split='valid')
    # iter_valid_loader = iter(valid_loader)
    # valid_wav, valid_singer = next(iter_valid_loader)


    print('training data shape: %s' % str(train_wav.shape))
    # print('valid data shape: %s' % str(valid_wav.shape))

    print(train_singer)


    # print('training data shape: %s' % str(train_wav.shape))