from transformers import TransfoXLConfig, TransfoXLModel
import torch

configuration = TransfoXLConfig()
model = TransfoXLModel(configuration)

x = torch.randint(0, configuration.vocab_size, (1, 512))
y = model(x)
output = y['last_hidden_state'] # outputshape = (batch, x_len, d_embed)


