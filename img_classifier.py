import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NNModel(nn.Module):
    def __init__(self, dict_length, isize, hsize, dsize, nlayers = 1):
        super(NNModel, self).__init__()
        self.dict_length = dict_length
        self.encoder = nn.LSTM(input_size = isize, hidden_size = hsize, num_layers = nlayers)
        self.decoder = nn.LSTM(input_size = self.dict_length, hidden_size = hsize, num_layers = nlayers)
        self.classify = nn.Linear(dsize, self.dict_length)

    def forward(self, inputs, caption_length):
        print(inputs.shape)
        _, (hn, cn) = self.encoder(inputs)
        # print(hn.shape)
        # print(cn.shape)
        col_list = torch.tensor(np.zeros((1,self.dict_length)), dtype=torch.float32)
        col_list[0,0] = 1
        d_output,(hn, cn) = self.decoder(col_list, (hn, cn))
        # print(d_output)
        col_list = torch.concatenate((col_list,d_output), axis=0)
        for n in range(caption_length-2):
            # print(n)
            d_output,(hn, cn) = self.decoder(d_output, (hn, cn))
            col_list = torch.concatenate((col_list,d_output), axis=0)
        # print(col_list.shape)

        return col_list



