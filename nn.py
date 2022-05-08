import torch
import torch.nn.functional as F


class TwoLayerNN(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(TwoLayerNN, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)
        self.dropout = torch.nn.Dropout(p=0.5)

        self.num_in_neurons = n_input
        self.num_hidden_neurons = n_hidden
        self.num_out_neurons = n_output

    def forward(self, x):
        h_input = self.hidden(x)
        h_output = torch.sigmoid(h_input)
        y_pred = self.out(h_output)

        return y_pred
