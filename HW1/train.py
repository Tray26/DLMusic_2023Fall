import torch
from torch import nn
from sklearn.metrics import accuracy_score, confusion_matrix
from model_cnn import CNN
from model_shortChunkCNN import ShortChunkCNN
import numpy as np
from dataloader import get_dataloader
import os
import csv



if __name__ == "__main__":
    support_data_path = './support_data'
    save_model_path = './models/'
    if not os.path.isdir(save_model_path):
        os.makedirs(save_model_path)
    with open(os.path.join(support_data_path, 'singers.csv'), newline='') as readsingers:
        singers_list = csv.reader(readsingers)
        singers_list = list(singers_list)
        singers_list = singers_list[0]
    num_classes = len(singers_list)
    # print(f'there are {num_classes} singers')
    # print(num_classes)
    # device_name = "mps" if torch.has_mps else "cpu"
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(device_name)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    select_net = 'shortchunkCNN'
    select_loss = 'BCE'
    # select_net = 'CNN'
    if select_net == 'shortchunkCNN':
        sample_interval = 3.69
        net = ShortChunkCNN(n_class=num_classes).to(device)
        # loss_function = nn.BCELoss()
    elif select_net == 'CNN':
        sample_interval = 29.1
        net = CNN(num_classes=num_classes).to(device)
        # loss_function = nn.CrossEntropyLoss()

    train_loader = get_dataloader(split='train', is_augmentation=True, sample_interval=sample_interval)
    valid_loader = get_dataloader(split='valid', is_augmentation=False, sample_interval=sample_interval)

    if select_loss == 'BCE':
        loss_function = nn.BCELoss()
    elif select_loss == 'CrossEntropy':
        loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    valid_losses = []
    num_epochs = 20

    best_valid_accuracy = 0
    best_epoch = 0

    best_valid_accuracy = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        # device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print(device_name)
        losses = []

        # Train
        net.train()
        for (wav, singer_index) in train_loader:
            wav = wav.to(device)
            
            # Forward
            # print(wav.shape)
            out = net(wav)
            # print(out.shape, singer_index.shape)
            # if loss_function == nn.BCELoss():
            if select_loss == 'BCE':
                label = torch.zeros(out.shape)
                intg = list(singer_index.shape)
                intg = int(intg[0])
                for index in range(intg):
                    label[index,singer_index[index]] = 1
                label = label.to(device)
                loss = loss_function(out, label)
            elif select_loss == 'CrossEntropy':
                singer_index = singer_index.to(device)
                loss = loss_function(out, singer_index)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('Epoch: [%d/%d], Train loss: %.4f' % (epoch+1, num_epochs, np.mean(losses)))

        # Validation
        net.eval()
        y_true = []
        y_pred = []
        losses = []
        for wav, singer_index in valid_loader:
            # print(type(wav), len(wav))
            wav = wav.to(device)
            # print(type(wav), wav.shape)
            # wav0 = wav.view()
            # singer_index = singer_index.to(device)

            # reshape and aggregate chunk-level predictions
            # b, c, t = wav.size()
            logits = net(wav)
            # logits = logits.mean(dim=1)
            if select_loss == 'BCE':
                label = torch.zeros(logits.shape)
                intg = list(singer_index.shape)
                intg = int(intg[0])
                for index in range(intg):
                    label[index,singer_index[index]] = 1
                label = label.to(device)
                loss = loss_function(logits, label)
            elif select_loss == 'CrossEntropy':
                singer_index = singer_index.to(device)
                loss = loss_function(logits, singer_index)

            # loss = loss_function(logits, singer_index)
            losses.append(loss.item())
            # logits = logits.detach().cpu()
            _, pred = torch.max(logits.data, 1)

            # append labels and predictions
            y_true.extend(singer_index.tolist())
            y_pred.extend(pred.tolist())
        accuracy = accuracy_score(y_true, y_pred)
        valid_loss = np.mean(losses)
        print('Epoch: [%d/%d], Valid loss: %.4f, Valid accuracy: %.4f' % (epoch+1, num_epochs, valid_loss, accuracy))

        # Save model
        valid_losses.append(valid_loss.item())
        # if np.argmin(valid_losses) == epoch:
        #     print('Saving the best model at %d epochs!' % epoch)
        #     torch.save(net.state_dict(), 'best_model.ckpt')

        if accuracy > best_valid_accuracy:
            print('Saving the best model at %d epochs!' % epoch)
            torch.save(net.state_dict(), f'{save_model_path}best_model_{select_net}_{select_loss}_{num_epochs}epochs.ckpt')
            # torch.save(net.state_dict(), 'best_model.ckpt')
            best_valid_accuracy = accuracy
            best_epoch = epoch

    print(f'Best Valid Accuracy: {best_valid_accuracy}, Get at Epoch {best_epoch}')    


