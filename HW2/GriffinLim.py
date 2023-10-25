import os 
import librosa
import glob

from Melspectrogram import mel_spectrogram, load_audio

num_mels=80
n_fft=2048
hop_size=256
win_size=1024
sampling_rate=48000
fmin=0
fmax=24000

if __name__ == '__main__':
    valid_path = './m4singer_valid'
    song_list = os.listdir(valid_path)
    for song in song_list:
        wav_list = sorted(glob.glob(os.path.join(valid_path, song, "*.wav")))
        mid_list = sorted(glob.glob(os.path.join(valid_path, song, "*.mid")))
        textGrid_list = sorted(glob.glob(os.path.join(valid_path, song, "*.TextGrid")))
        for wav in wav_list:
            # audio_path = os.path.join(valid_path, song, wav)
            audio = load_audio(audio_path=wav)
            print(audio.shape)
            # mel_tensor = mel_spectrogram(
            #     audio, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax
            # )
            # mel = mel_tensor.squeeze().cpu().numpy()

            # print(mel.shape)

            # recon_audio = librosa.feature.inverse.mel_to_audio(
            #     mel, sr=sampling_rate, n_fft=n_fft,
            #     hop_length=hop_size, win_length=win_size
            # )



    # print(song_list)


