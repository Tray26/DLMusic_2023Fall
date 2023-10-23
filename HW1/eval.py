import os
import csv
import numpy as np
import torch
from testloader import get_testloader
from dataloader_2 import _get_dataloader
from model_shortChunkCNN import ShortChunkCNN
from read_singers import get_singers
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import cv2
import matplotlib.pyplot as plt
# import seaborn as sns



if __name__ == "__main__":
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(device_name)
    net = ShortChunkCNN().to(device)

    batch_size = 16
    num_samples = 59049

    test_loader = get_testloader()
    valid_loader = _get_dataloader(split='valid', num_samples=num_samples, batch_size=batch_size)
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
            logits = net(wav)
            song_id = int(song_id.numpy())

            total_points = torch.sum(logits, dim=0)
            _, top3_indices = torch.topk(total_points, k=3)
            top3_indices = top3_indices.cpu().numpy().astype(np.uint16)
            
            top_singers_list = [song_id]
            for i in range(3):
                top_singers_list.append(singers_list[top3_indices[i]])

            if len(id_list) == 0:
                data_list.append(top_singers_list)
                id_list.append(song_id)
            else:
                all_less = [id for id in id_list if id < song_id]
                insert_index = len(all_less)
                id_list.insert(insert_index, song_id)
                data_list.insert(insert_index, top_singers_list)
        y_true = []
        y_pred = []
        for wav, singer_index in valid_loader:
            # print('ok')
            wav = torch.squeeze(wav)
            wav = wav.to(device)
            
            logits = net(wav)
            singer_index = singer_index.repeat(batch_size)

            # if select_loss == 'BCE':
            #     label = torch.zeros(logits.shape)
            #     intg = list(singer_index.shape)
            #     intg = int(intg[0])
            #     for index in range(intg):
            #         label[index,singer_index[index]] = 1
            #     label = label.to(device)
            #     loss = loss_function(logits, label)
            # elif select_loss == 'CrossEntropy':
            #     singer_index = singer_index.to(device)
            #     loss = loss_function(logits, singer_index)

            # losses.append(loss.item())
            logits = logits.detach().cpu()
            _, pred = torch.max(logits.data, 1)

            # append labels and predictions
            y_true.extend(singer_index.tolist())
            y_pred.extend(pred.tolist())
        
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        # sns.heatmap(cm, annot=True, xticklabels=GTZAN_GENRES, yticklabels=GTZAN_GENRES, cmap='YlGnBu')
        print('Accuracy: %.4f' % accuracy)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=singers_list)
        print(type(cm_display))

        cm_filename = './SaveConfusionMtrix.png'
        # cv2.imwrite(cm_filename, cm_display)
        plt.imsave(cm_filename, cm_display)



    # with open('r12942144.csv', 'w', newline='') as csvifle:
    #     csvwriter = csv.writer(csvifle)
    #     csvwriter.writerows(data_list)

            

