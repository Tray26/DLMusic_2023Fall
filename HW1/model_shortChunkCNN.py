from torch import nn
import torchaudio
from modules import Conv_1d, ResSE_1d, Conv_2d, Res_2d, Conv_V, Conv_H, Res_2d_mp
# , HarmonicSTFT

class ShortChunkCNN(nn.Module):
    '''
    Short-chunk CNN architecture.
    So-called vgg-ish model with a small receptive field.
    Deeper layers, smaller pooling (2x2).
    '''
    def __init__(self,
                n_channels=128,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=128,
                n_class=20):
        super(ShortChunkCNN, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Conv_2d(1, n_channels, pooling=2)
        self.layer2 = Conv_2d(n_channels, n_channels, pooling=2)
        self.layer3 = Conv_2d(n_channels, n_channels*2, pooling=2)
        self.layer4 = Conv_2d(n_channels*2, n_channels*2, pooling=2)
        self.layer5 = Conv_2d(n_channels*2, n_channels*2, pooling=2)
        self.layer6 = Conv_2d(n_channels*2, n_channels*2, pooling=2)
        self.layer7 = Conv_2d(n_channels*2, n_channels*4, pooling=2)

        # Dense
        self.dense1 = nn.Linear(n_channels*4, n_channels*4)
        self.bn = nn.BatchNorm1d(n_channels*4)
        self.dense2 = nn.Linear(n_channels*4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        print(x.shape)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)

        return x


