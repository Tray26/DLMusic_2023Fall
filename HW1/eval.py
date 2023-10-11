import os
import csv
import numpy as np
import torch
from testloader import get_testloader
from model_shortChunkCNN import ShortChunkCNN
from read_singers import get_singers


if __name__ == "__main__":
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(device_name)
    net = ShortChunkCNN().to(device)
    test_loader = get_testloader()
    best_model = 'best_model_short_BCE_400epochs.ckpt'
    singers_list = get_singers()

    S = torch.load(os.path.join('./models', best_model))
    net.load_state_dict(S)
    print('loaded')

    net.eval()

    data_list = []
    id_list = []
    

    with torch.no_grad():
        for wav, song_id in test_loader:
            wav = torch.squeeze(wav)
            wav = wav.to(device)
            # song_id = song_id.to(device)
            logits = net(wav)
            song_id = int(song_id.numpy())


            print(song_id)

            total_points = torch.sum(logits, dim=0)
            _, top3_indices = torch.topk(total_points, k=3)
            top3_indices = top3_indices.cpu().numpy().astype(np.uint16)
            
            top_singers_list = [song_id]
            for i in range(3):
                top_singers_list.append(singers_list[top3_indices[i]])
            # top3_names = singers_list[top3_indices]

            # print(top3_indices)
            # print(top_singers_list)

            if len(id_list) == 0:
                data_list.append(top_singers_list)
                id_list.append(song_id)
            else:
                all_less = [id for id in id_list if id < song_id]
                insert_index = len(all_less)
                id_list.insert(insert_index, song_id)
                data_list.insert(insert_index, top_singers_list)

    with open('r12942144.csv', 'w', newline='') as csvifle:
        csvwriter = csv.writer(csvifle)
        csvwriter.writerows(data_list)

            

