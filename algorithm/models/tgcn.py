import torch
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np

class TGCNCell(torch.nn.Module):
    def __init__(self, args):
        super(TGCNCell, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.num_nodes = args.input_size
        self.graph_conv1 = GCNConv(self.hidden_size+self.input_size, self.hidden_size * 2)
        self.graph_conv2 = GCNConv(self.hidden_size+self.input_size, self.hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.graph_conv1.bias, 1.0)

    def forward(self, x, edge_index, hidden_state):

        ru_input = torch.concat([x, hidden_state], dim=1)

        ru = torch.sigmoid(self.graph_conv1(ru_input, edge_index))
        r, u = torch.chunk(ru.reshape([-1, self.num_nodes * 2 * self.hidden_size]), chunks=2, dim=1)
        r = r.reshape([-1, self.hidden_size])
        u = u.reshape([-1, self.hidden_size])

        c_input = torch.concat([x, r * hidden_state], dim=1)
        c = torch.tanh(self.graph_conv2(c_input, edge_index))

        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state

def load_edge_index_from_csv(file_path):
    adjacency_matrix = pd.read_csv(file_path, header=None).values
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float)
    edge_index = adjacency_matrix.nonzero(as_tuple=False).t().contiguous()
    return edge_index
