import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from body.body_wrapper import TrainedBody
from data.streaming import RawStream
from util import mu_law_encoding


class LSMGRU(nn.Module):
    def __init__(self, hqa, inp_dim, out_dim, hidden_dim, num_layers):
        super().__init__()
        self.hqa = hqa
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=inp_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # GRU expects x to be of shape batch_size,seq_lenth,feat_num
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        return x.permute(0, 2, 1)

    def get_feats(self, x):
        """ Get features to be used in downstream tasks """
        x = self.hqa.encode(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        return x


class TrainedLSM(TrainedBody):
    def __init__(self, lsm_model):
        feat_dim = lsm_model.hidden_dim
        super().__init__(feat_dim=feat_dim, data_class=RawStream)
        self.lsm_model = lsm_model

    def forward(self, inputs):
        with torch.no_grad():
            x_mu = mu_law_encoding(inputs).unsqueeze(1)
            x = self.lsm_model.get_feats(x_mu)
        return x


class LSMTransformer(nn.Module):
    def __init__(self, inp_dim, out_dim, num_heads, hidden_dim, num_layers, dropout=0.0):
        super().__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(inp_dim, dropout)
        encoder_layers = TransformerEncoderLayer(inp_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(inp_dim, out_dim)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # Transformer expects src to be of shape seq_length,batch_size,feat_num
        src = src.permute(2, 0, 1)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        output = output.permute(1, 2, 0)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=25000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
