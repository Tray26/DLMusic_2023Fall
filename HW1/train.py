import torch
from torch import nn
from sklearn.metrics import accuracy_score, confusion_matrix
from model_cnn import CNN
import numpy as np
from dataloader import get_dataloader



if __name__ == "__main__":
    train_loader = get_dataloader(split='train', is_augmentation=True)
    valid_loader = get_dataloader(split='valid', is_augmentation=True)

    # device_name = "mps" if torch.has_mps else "cpu"
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(device_name)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cnn = CNN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    valid_losses = []
    num_epochs = 30

    for epoch in range(num_epochs):
        # device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print(device_name)
        losses = []

        # Train
        cnn.train()
        for (wav, singer_index) in train_loader:
            wav = wav.to(device)
            singer_index = singer_index.to(device)

            # Forward
            out = cnn(wav)
            loss = loss_function(out, singer_index)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('Epoch: [%d/%d], Train loss: %.4f' % (epoch+1, num_epochs, np.mean(losses)))

        # Validation
        cnn.eval()
        y_true = []
        y_pred = []
        losses = []
        for wav, singer_index in valid_loader:
            # print(type(wav), len(wav))
            wav = wav.to(device)
            # print(type(wav), wav.shape)
            # wav0 = wav.view()
            singer_index = singer_index.to(device)

            # reshape and aggregate chunk-level predictions
            # b, c, t = wav.size()
            logits = cnn(wav)
            # logits = logits.mean(dim=1)
            loss = loss_function(logits, singer_index)
            losses.append(loss.item())
            _, pred = torch.max(logits.data, 1)

            # append labels and predictions
            y_true.extend(singer_index.tolist())
            y_pred.extend(pred.tolist())
        accuracy = accuracy_score(y_true, y_pred)
        valid_loss = np.mean(losses)
        print('Epoch: [%d/%d], Valid loss: %.4f, Valid accuracy: %.4f' % (epoch+1, num_epochs, valid_loss, accuracy))

        # Save model
        valid_losses.append(valid_loss.item())
        if np.argmin(valid_losses) == epoch:
            print('Saving the best model at %d epochs!' % epoch)
            torch.save(cnn.state_dict(), 'best_model.ckpt')
