import os
import torch
import numpy as np
import soundfile as sf
from torch.utils import data
import csv
import numpy as np
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

class singerDataset(data.Dataset):
    def __init__(self, data_path, split, num_samples, sample_rate, batch_size, augmentation,
                 support_data_path = './support_data') -> None:
        super().__init__()
        self.data_path = data_path
        self.support_data_path = support_data_path
        self.split = split
        self.num_samples = int(np.floor(num_samples))
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.is_augmentation = augmentation
        self.get_songlist()
        with open(os.path.join(support_data_path, 'singers.csv'), newline='') as readsingers:
            singers_list = csv.reader(readsingers)
            singers_list = list(singers_list)
            singers_list = singers_list[0]
        self.singers = singers_list                                 #singers list
        if augmentation:
            self._get_augmentations()
    
    # get all somg info
    def get_songlist(self):
        list_songinfo_path = os.path.join(self.support_data_path, '%s.txt' % self.split)
        with open(list_songinfo_path) as f:
            lines = f.readlines()
        self.song_infolist = lines

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

    def get_audio(self, index):
        song_info = self.song_infolist[index].split(', ')
        
        song_path, singer_name, song_name = song_info[0], song_info[1], song_info[2]

        wav, fs = sf.read(song_path)
        if len(wav) < self.num_samples:
            # print('Audio not long enough:', line.split(', ')[2])
            len_mul = self.num_samples // len(wav)
            wav = np.repeat(wav, len_mul + 1)
        singer_index = self.singers.index(singer_name)

        return wav, singer_index
    
    def __getitem__(self, index):
        wav, singer_index = self.get_audio(index)
        if self.split == 'train':
            random_index = int(np.floor(np.random.random(1) * (len(wav)-self.num_samples)))
            wav = wav[random_index:random_index+self.num_samples]
            wav = wav.astype('float32')
            if self.is_augmentation:
                wav = self.augmentation(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()
            return wav, singer_index
        else:
            length = len(wav)
            hop = (length - self.num_samples) // self.batch_size
            # print(hop)
            x = torch.zeros(self.batch_size, self.num_samples)
            # print(x.shape)
            for i in range(self.batch_size):
                x[i] = torch.Tensor(wav[i*hop:i*hop+self.num_samples]).unsqueeze(0)
            x = x.numpy().astype('float32')

            return x, singer_index

        


    def __len__(self):
        return len(self.song_infolist)

def _get_dataloader(data_path='./artist20/mp3s-32k/',
                    split='train',
                    num_samples = 59049,
                    sample_rate=16000,
                    batch_size=16,
                    num_workers=4,
                    ):
    is_shuffle = True if (split == 'train') else False
    real_batch_size = batch_size if (split == 'train') else (1)
    # augmentation = True if (split == 'train') else False
    augmentation = False
    data_loader = data.DataLoader(dataset=singerDataset(data_path, 
                                                        split,
                                                        num_samples,
                                                        sample_rate,
                                                        batch_size,
                                                        augmentation),
                                batch_size=real_batch_size,
                                shuffle=is_shuffle,
                                drop_last=True,
                                num_workers=num_workers)
    return data_loader

if __name__ == "__main__":
    train_loader = _get_dataloader(split='train')
    iter_train_loader = iter(train_loader)
    train_wav, train_singer = next(iter_train_loader)

    valid_loader = _get_dataloader(split='valid')
    iter_valid_loader = iter(valid_loader)
    valid_wav, valid_singer = next(iter_valid_loader)

    valid_singer = valid_singer.repeat(3)

    print('training data shape: %s' % str(train_wav.shape))
    print('valid data shape: %s' % str(valid_wav.shape))

    print(train_singer)
    print(valid_singer)

