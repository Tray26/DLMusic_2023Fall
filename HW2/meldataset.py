import os
import random
import torch 
import torchaudio
import glob
import numpy as np
from torch.utils.data import DataLoader, Dataset

from Melspectrogram import mel_spectrogram, load_audio
from spectrogram_config import readSpecConfig

class MelDataset(Dataset):
    def __init__(self, data_path, sampling_period, real_batch_size, segments_per_song,
                 split = 'train', spec_config_path='./spec_config.csv') -> None:
        super().__init__()
        self.data_path = data_path
        self.sampling_period = sampling_period
        self.real_batch_size = real_batch_size
        self.segments_per_song = segments_per_song

        self.spec_config = readSpecConfig(spec_config_path, split)
        self.num_mels = self.spec_config['num_mels']
        self.n_fft = self.spec_config['n_fft']
        self.hop_size = self.spec_config['hop_size']
        self.win_size = self.spec_config['win_size']
        self.sampling_rate = self.spec_config['sampling_rate']
        self.fmin = self.spec_config['fmin']
        self.fmax = self.spec_config['fmax']

        self.get_song_list()
        self.get_files()
        # self.sample_num = int()
        
    def get_song_list(self):
        self.song_list = os.listdir(self.data_path)

    def get_files(self):
        self.wav_list = []
        self.mid_list = []
        self.textGrid_list = []
        for song in self.song_list:
        # song = self.song_list[index]
            wav_list = glob.glob(os.path.join(self.data_path, song, "*.wav"))
            mid_list = glob.glob(os.path.join(self.data_path, song, "*.mid"))
            textGrid_list = glob.glob(os.path.join(self.data_path, song, "*.TextGrid"))

            self.wav_list.append(wav_list)
            self.mid_list.append(mid_list)
            self.textGrid_list.append(textGrid_list)
        # return wav_list, mid_list, textGrid_list

    def get_audio_segment(self, wav_path):
        sample_num = int(self.sampling_period * self.sampling_rate)
        raw_wav = load_audio(audio_path=wav_path)
        random_index = int(np.floor(np.random.random(1) * (raw_wav.shape[1]-sample_num)))
        wav_segment = raw_wav[:, random_index:random_index+sample_num]
        # wav_segment = wav_segment.astype('float32')

        self.sample_num = sample_num

        return wav_segment
    
    def __getitem__(self, index):
        wav_path = self.wav_list[index]
        wav_segment = self.get_audio_segment(wav_path)
        mel_tensor = mel_spectrogram(
            wav_segment, self.n_fft, self.num_mels, self.sampling_rate, 
            self.hop_size, self.win_size, self.fmin, self.fmax
        )
        mel = mel_tensor.squeeze().cpu().numpy()
        # wav_list, _, _ = self.get_files(index)
        # sample_wavs = random.sample(wav_list, self.segments_per_song)
        # for wav_path in sample_wavs:
        #     wav_segment = self.get_audio_segment(wav_path)
            

        return wav_segment, mel
    

    def __len__(self):
        return len(self.wav_list)

        




