import os
import torch
import numpy as np
import soundfile as sf
from torch.utils import data
import csv
import numpy as np

from read_singers import get_singers

class testDataset(data.Dataset):
    def __init__(self, data_path, num_samples, sample_rate, batch_size,) -> None:
        super().__init__()
        self.data_path = data_path
        self.num_samples = int(np.floor(num_samples))
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.singers = get_singers()
        self.songlist = os.listdir(data_path)

    def get_audio(self, index):
        song_path = os.path.join(self.data_path, self.songlist[index])
        wav, _ = sf.read(song_path)
        return wav

    def __getitem__(self, index):
        # song index
        song_filename = self.songlist[index]
        song_id = song_filename.split('.')[0]
        song_id = int(song_id)

        # audio
        raw_audio = self.get_audio(index)
        audio_length = len(raw_audio)
        hop = (audio_length - self.num_samples) // self.batch_size
        wav = torch.zeros(self.batch_size, self.num_samples)
        for i in range(self.batch_size):
            wav[i] = torch.Tensor(raw_audio[i*hop:i*hop+self.num_samples]).unsqueeze(0)
        return wav, song_id
    
    def __len__(self):
        return len(self.songlist)
    
def get_testloader(data_path='./artist20_testing_data',
                    num_samples = 59049,
                    sample_rate=16000,
                    real_batch_size=16,
                    num_workers=4):
    dataloader_batchsize = 1
    testLoader = data.DataLoader(dataset=testDataset(data_path=data_path,
                                                     num_samples=num_samples,
                                                     sample_rate=sample_rate,
                                                     batch_size=real_batch_size),
                                    batch_size=dataloader_batchsize,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=num_workers)
    return testLoader


if __name__ == "__main__":
    test_loader = get_testloader()
    iter_test_loader = iter(test_loader)
    test_wav = next(iter_test_loader)

    print('training data shape: %s' % str(test_wav.shape))
    # print(tes)

