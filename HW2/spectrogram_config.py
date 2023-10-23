import os
import csv
import json

num_mels=80
n_fft=2048
hop_size=256
win_size=1024
sampling_rate=48000
fmin=0
fmax=24000

def readSpecConfig(config_path, split):
    with open(config_path, 'r', newline='') as dictfile:
        reader = csv.DictReader(dictfile)
        dict_list = list(reader)
    return next((item for item in dict_list if item['split'] == split), None)


if __name__ == '__main__':
    train_config = dict(
        split = 'train',
        num_mels = num_mels,
        n_fft=n_fft,
        hop_size=hop_size,
        win_size=win_size,
        sampling_rate=sampling_rate,
        fmin=fmin,
        fmax=fmax
    )
    valid_config = dict(
        split = 'valid',
        num_mels = num_mels,
        n_fft=n_fft,
        hop_size=hop_size,
        win_size=win_size,
        sampling_rate=sampling_rate,
        fmin=fmin,
        fmax=fmax
    )
    test_config = dict(
        split = 'test',
        num_mels = num_mels,
        n_fft=n_fft,
        hop_size=hop_size,
        win_size=win_size,
        sampling_rate=22050,
        fmin=fmin,
        fmax=11025
    )
    
    train_json = json.dumps(train_config)
    valid_json = json.dumps(valid_config)
    test_json = json.dumps(test_config)
    with open('./train_spec_config.json', 'w') as outfile:
        outfile.write(train_json)
    with open('./valid_spec_config.json', 'w') as outfile:
        outfile.write(valid_json)
    with open('./test_spec_config.json', 'w') as outfile:
        outfile.write(test_json)

    spec_config_path = './spec_config.csv'

    with open(spec_config_path, 'w', newline='') as writedict:
        writer = csv.DictWriter(writedict, fieldnames=train_config.keys())
        writer.writeheader()
        writer.writerow(train_config)
        writer.writerow(valid_config)
        writer.writerow(test_config)
    # w = csv.DictWriter(open("tmp.csv", "w", newline=''), fieldnames=train_config.keys())
    # w.writeheader()
    # w.writerow(train_config)
    # w.writerow(test_config)

    # for row in csv.DictReader(open('tmp.csv', 'r')):
    #     print(row)

    # with open('./spec_config.csv', 'r', newline='') as dictfile:
    #     reader = csv.DictReader(dictfile)
    #     dict_list = list(reader)
    #     for row in reader:
    #         print(row['split'])
    #     # print(dict_list)
    # train_spec = next((item for item in dict_list if item['split'] == "train"), None)

    train_spec = readSpecConfig(spec_config_path, 'train')
    print(train_spec)
    valid_spec = readSpecConfig(spec_config_path, 'valid')
    test_spec = readSpecConfig(spec_config_path, 'test')
    print(valid_spec)
    print(test_spec)

