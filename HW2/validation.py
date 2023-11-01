import itertools
import os
import time
import argparse
import json
import torch
from scipy.io.wavfile import write
from torch.utils.data import DistributedSampler, DataLoader
import torch.nn.functional as F
from hifigan.models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from hifigan.env import AttrDict, build_env
from hifigan.utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
from hifigan.meldataset import MelDataset, mel_spectrogram, get_dataset_filelist

def main():
    print('Initializing Validation Process..')

    parser = argparse.ArgumentParser()

    # parser.add_argument('--group_name', default=None)
    # parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    # parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_filepath', default='./m4singer')
    parser.add_argument('--input_validation_filepath', default='./m4singer_valid')
    parser.add_argument('--checkpoint_path', default='./checkpoints/cp_hifigan_finetune_1')
    parser.add_argument('--config', default='./hifigan/config_v1.json')

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(0))

    generator = Generator(h).to(device)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')

    state_dict_g = load_checkpoint(cp_g, device)
    generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    torch.cuda.empty_cache()
    val_err_tot = 0

    _, validation_filelist = get_dataset_filelist(a)
    validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device)
    validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                    sampler=None,
                                    batch_size=1,
                                    pin_memory=True,
                                    drop_last=True)
    
    audio_save_path = './recon_results/hifigan_finetune'
    if not os.path.isdir(audio_save_path):
        os.makedirs(audio_save_path)

    with torch.no_grad():
        for j, batch in enumerate(validation_loader):
            x, y, filename, y_mel = batch
            y_g_hat = generator(x.to(device))
            recon_audio = y_g_hat.cpu().numpy()
            # print(type(y_g_hat.cpu().numpy()))
            song_name, wav_name = filename[0].split('/')[-2], filename[0].split('/')[-1]
            # print(song_name, wav_name)

            # song_dir_path = os.path.join(audio_save_path, song_name)
            # if not os.path.isdir(song_dir_path):
            #     os.makedirs(song_dir_path)

            audio_path = os.path.join(audio_save_path, song_name+'%'+ wav_name)
            write(audio_path, h.sampling_rate, recon_audio)
            print(f'Done writing {audio_path}')

            # y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            # y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
            #                                 h.hop_size, h.win_size,
            #                                 h.fmin, h.fmax_for_loss)
            # val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

        #     if j <= 4:
        #         if steps == 0:
        #             sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
        #             sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

        #         sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
        #         y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
        #                                         h.sampling_rate, h.hop_size, h.win_size,
        #                                         h.fmin, h.fmax)
        #         sw.add_figure('generated/y_hat_spec_{}'.format(j),
        #                         plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

        # val_err = val_err_tot / (j+1)
        # sw.add_scalar("validation/mel_spec_error", val_err, steps)

if __name__ == '__main__':
    main()
