import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv

class TGCN(nn.Module):
    def __init__(self, args, device):
        super(TGCN, self).__init__()
        self.hidden_size = args.hidden_size
        self.gru_cell = nn.GRUCell(args.input_size, args.hidden_size)
        self.args = args
        self.device = device
        self.fc = nn.Linear(args.hidden_size, args.input_size)

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
        for _ in range(1, self.args.pre_len):
            h = self.gru_cell(out, h)  # h: (batch_size, hidden_size)
            out = self.fc(h)  # out: (batch_size, input_size)
            pred = torch.cat((pred, out.unsqueeze(1)), dim=1)  # pred: (batch_size, pre_len, input_size)
        return pred
    
class TGCNCell(torch.nn.Module):
    def __init__(self, args):
        super(TGCNCell, self).__init__()
        self.args = args
        self.num_features = args.c_in
        self.nhid = args.c_out
        self.seq_len = args.seq_len
        self.num_nodes = args.num_nodes

        # 这是仿照作者源码里的写法，实际上这是两个GCN，在forward函数中会将其输出拆成两半
        self.graph_conv1 = GCNConv(self.nhid+self.num_features, self.nhid * 2)
        self.graph_conv2 = GCNConv(self.nhid+self.num_features, self.nhid)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.graph_conv1.bias, 1.0)

    def forward(self, x, edge_index, hidden_state):

        ru_input = torch.concat([x, hidden_state], dim=1)

        # 这里将一个GCN的输出拆成两半，如果熟悉其矩阵写法的话，实际上就是用了俩GCN
        # 但是这里的拆分函数也是仿照源码，个人觉得拆分的维度不对，但是这么写的准确率高
        ru = torch.sigmoid(self.graph_conv1(ru_input, edge_index))
        r, u = torch.chunk(ru.reshape([-1, self.num_nodes * 2 * self.nhid]), chunks=2, dim=1)
        r = r.reshape([-1, self.nhid])
        u = u.reshape([-1, self.nhid])

        c_input = torch.concat([x, r * hidden_state], dim=1)
        c = torch.tanh(self.graph_conv2(c_input, edge_index))

        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state