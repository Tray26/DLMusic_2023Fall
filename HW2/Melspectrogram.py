import numpy as np
import torch
import torchaudio
import librosa
import os
import matplotlib.pyplot as plt
import glob
from scipy.io.wavfile import write

num_mels=80
n_fft=1024
hop_size=256
win_size=1024
sampling_rate=22050
fmin=0
fmax=8000

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    device = y.device
    melTorch = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate, n_fft=n_fft, n_mels=num_mels, \
           hop_length=hop_size, win_length=win_size, f_min=fmin, f_max=fmax, pad=int((n_fft-hop_size)/2), center=center).to(device)      
    spec = melTorch(y)
    spec = spectral_normalize_torch(spec)
    return spec

def to_mono(audio, dim=-2): 
    if len(audio.size()) > 1:
        return torch.mean(audio, dim=dim, keepdim=True)
    else:
        return audio

def load_audio(audio_path, sr=None, mono=True):
    if 'mp3' in audio_path:
        torchaudio.set_audio_backend('sox_io')
    audio, org_sr = torchaudio.load(audio_path)
    audio = to_mono(audio) if mono else audio
    
    if sr and org_sr != sr:
        audio = torchaudio.transforms.Resample(org_sr, sr)(audio)

    return audio

if __name__ == '__main__':
    load_audio_path = './m4singer_valid'
    save_npy_path = './m4singer_mel'
    new_audio_path = './m4singer_1'
    if not os.path.exists(save_npy_path):
        os.mkdir(save_npy_path)
    if not os.path.exists(new_audio_path):
        os.mkdir(new_audio_path)
    song_list = os.listdir(load_audio_path)
    audio_list = []
    for song in song_list:
        # print(song)
        wav_list = glob.glob(os.path.join(load_audio_path, song, "*.wav"))
        # wav_list = os.listdir(os.path.join(load_audio_path, song))
        # print(wav_list)
        audio_list.extend(wav_list)
    # print(audio_list)
    # audio_list = os.listdir(load_audio_path)
    audio_list.sort()
    for audio in audio_list:
        y = load_audio(audio, sr=sampling_rate)
        # print(type(y))
        mel_tensor = mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax)
        mel = mel_tensor.squeeze().cpu().numpy()
        # wav_path_list = audio.split('/')
        song_name, wav_name = audio.split('/')[-2], audio.split('/')[-1]
        # print(song_name, wav_name)
        file_name = os.path.join(save_npy_path, song_name+'%'+wav_name[:-4]+'.npy')
        # print(file_name)
        y_np = y.squeeze().cpu().numpy()
        audio_save_path = os.path.join(new_audio_path, song_name+'%'+wav_name)
        # print(audio_save_path)
        write(audio_save_path, sampling_rate, y_np)
        print(f'Done writing {audio_save_path}')
        # song_mel_dir = os.path.join(save_npy_path, song_name)
        # if not os.path.isdir(song_mel_dir):
        #     os.makedirs(song_mel_dir)
        np.save(file_name, mel)
        print(f'Done writing {file_name}')
        mel = np.load(file_name) # check the .npy is readable

    # plot the last melspectrogram
    # ref: https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # don't forget to do dB conversion
    # S_dB = librosa.power_to_db(mel, ref=np.max)
    # img = librosa.display.specshow(S_dB, x_axis='time',
    #                          y_axis='mel', sr=sampling_rate,
    #                          fmax=fmax, ax=ax, hop_length=hop_size, n_fft=n_fft)
    # fig.colorbar(img, ax=ax, format='%+2.0f dB')
    # ax.set(title='Mel-frequency spectrogram')