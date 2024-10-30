import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, pre_len, device):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.pre_len = pre_len
        self.device = device
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, h=None):
        # x: (batch_size, seq_len, input_size)
        # h: (1, batch_size, hidden_size)
        batch_size, seq_len, input_size = x.shape
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size).to(self.device)  # h: (batch_size, hidden_size)
        for i in range(seq_len):
            h = self.gru_cell(x[:, i, :], h)  # h: (batch_size, hidden_size)
        out = self.fc(h)  # out: (batch_size, input_size)
        pred = out.unsqueeze(1)  # pred: (batch_size, 1, input_size)
        for _ in range(1, self.pre_len):
            h = self.gru_cell(out, h)  # h: (batch_size, hidden_size)
            out = self.fc(h)  # out: (batch_size, input_size)
            pred = torch.cat((pred, out.unsqueeze(1)), dim=1)  # pred: (batch_size, pre_len, input_size)
        return pred


