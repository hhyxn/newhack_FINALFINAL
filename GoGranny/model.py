import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchtext

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(LSTM, self).__init__()
        #making one hot encodings for the input
        self.emb = torch.eye(input_size)
        self.hidden_size = hidden_size
        #this is the LSTM model
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        #our fc layers
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_class)

    def forward(self, x, lengths):
        # Look up the embedding (one-hot encoding)
        x = self.emb[x]

        # Pack the sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Set the initial hidden states
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate the RNN
        packed_output, _ = self.rnn(packed_input, (h0, c0))

        # Unpack the sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Concatenate max and mean pooling of the output
        out_max, _ = torch.max(output, dim=1)
        out_mean = torch.mean(output, dim=1)
        out = torch.cat((out_max, out_mean), dim=1)

        # Pass through the fully connected layer
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out