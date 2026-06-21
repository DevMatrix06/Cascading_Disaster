import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class CascadeGNN(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=64, out_channels=1, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        return torch.sigmoid(x).squeeze(-1)


class TemporalCascadeGNN(nn.Module):
    def __init__(self, static_in=6, temporal_in=3, lstm_hidden=32,
                 hidden_channels=64, out_channels=1, dropout=0.3):
        super().__init__()
        # temporal encoder: LSTM over (num_months, temporal_in)
        self.lstm = nn.LSTM(
            input_size=temporal_in,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        # GNN layers take static features + LSTM last hidden state
        sage_in = static_in + lstm_hidden
        self.conv1 = SAGEConv(sage_in, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, temporal_x, edge_index):
        # temporal_x: (num_nodes, num_months, temporal_in)
        _, (h_n, _) = self.lstm(temporal_x)
        temporal_emb = h_n.squeeze(0)          # (num_nodes, lstm_hidden)

        combined = torch.cat([x, temporal_emb], dim=-1)  # (num_nodes, static_in + lstm_hidden)

        combined = self.conv1(combined, edge_index)
        combined = F.relu(combined)
        combined = F.dropout(combined, p=self.dropout, training=self.training)

        combined = self.conv2(combined, edge_index)
        combined = F.relu(combined)
        combined = F.dropout(combined, p=self.dropout, training=self.training)

        combined = self.conv3(combined, edge_index)
        return torch.sigmoid(combined).squeeze(-1)
