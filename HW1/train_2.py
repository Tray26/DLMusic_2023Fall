import torch
from torch import nn
from sklearn.metrics import accuracy_score, confusion_matrix
from model_cnn import CNN
from model_shortChunkCNN import ShortChunkCNN
from model_FCN import FCN
import numpy as np
from dataloader_2 import _get_dataloader
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
    print(f'there are {num_classes} singers')
    # device_name = "mps" if torch.has_mps else "cpu"
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(device_name)

    batch_size = 16

    # select_net = 'CNN'
    select_net = 'short'
    # select_net = 'FCN'
    select_loss = 'BCE'
    # select_loss = 'CrossEntropy'
    if select_net == 'short':
        num_samples = 59049
        net = ShortChunkCNN(n_class=num_classes).to(device)
    elif select_net == 'CNN':
        num_samples = 16000 * 29.1
        net = CNN(num_classes=num_classes).to(device)
    elif select_net == 'FCN':
        num_samples = 16000 * 29.1
        net = FCN(n_class=num_classes).to(device)

    if select_loss == 'BCE':
        loss_function = nn.BCELoss()
    elif select_loss == 'CrossEntropy':
        loss_function = nn.CrossEntropyLoss()

    train_loader = _get_dataloader(split='train', num_samples=num_samples, batch_size=batch_size)
    valid_loader = _get_dataloader(split='valid', num_samples=num_samples, batch_size=batch_size)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    valid_losses = []
    num_epochs = 100
    
    
    
    best_valid_accuracy = 0
    best_epoch = 0

    for epoch in range(num_epochs):
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

        net.eval()
        y_true = []
        y_pred = []
        losses = []
        for wav, singer_index in valid_loader:
            # print('ok')
            wav = torch.squeeze(wav)
            wav = wav.to(device)
            
            logits = net(wav)
            singer_index = singer_index.repeat(batch_size)

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

    print(f'Best Valid Accuracy: {best_valid_accuracy}, Get at Epoch {best_epoch+1}')    



