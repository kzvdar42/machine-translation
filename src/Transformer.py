import torch
from torch import nn, Tensor
import math



class PositionalEncoding(nn.Module):
    # From https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
    
    def create_pe(self, seq_len):
        pe = torch.zeros(seq_len, self.d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def forward(self, x):
        pe = self.create_pe(x.size(0))
        x = x + pe.to(x.device)
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntokens_src, ntokens_tgt, ninp, nhead, dim_feedforward, nlayers, pad_token, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import Transformer
        self.model_type = 'Transformer'
        self.ninp = ninp
        self.pad_token = pad_token
        self.masks = {
            'src': None,
            'tgt': None,
            'memory': None,
        }
        # Token Encoders
        self.src_encoder = nn.Embedding(ntokens_src, ninp)
        self.tgt_encoder = nn.Embedding(ntokens_tgt, ninp)
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        # Transformer
        self.transformer = Transformer(
            d_model=ninp,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
        )
        self.out = nn.Linear(ninp, ntokens_tgt)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sx, sy=None):
        """Generate matrix for seqential reveal of tokens."""
        sy = sy or sx
        mask = (torch.triu(torch.ones((sx, sy))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        self.transformer._reset_parameters()
    
    def preprocess(self, x, x_type):
        # Create masks
        padding_mask = (x == self.pad_token).bool().t()
        if self.masks[x_type] is None or self.masks[x_type].size(0) != len(x):
            self.masks[x_type] = self._generate_square_subsequent_mask(len(x), len(x)).to(x.device)
        
        x_enc = self.src_encoder(x) if x_type == 'src' else self.tgt_encoder(x)
        x_enc *= math.sqrt(self.ninp)
        x_enc = self.pos_encoder(x_enc)
        
        return x_enc, self.masks[x_type], padding_mask
        
    def forward(self, src, tgt):

        # TODO: Do we need memory mask?
        if (    self.masks['memory'] is None or
                self.masks['src'].size(0) != len(src) or
                self.masks['tgt'].size(0) != len(tgt)):
            self.masks['memory'] = self._generate_square_subsequent_mask(len(src), len(tgt)).to(src.device)
        
        src_enc, _, src_key_padding_mask = self.preprocess(src, 'src')
        tgt_enc, _, tgt_key_padding_mask = self.preprocess(tgt, 'tgt')
        memory_key_padding_mask = src_key_padding_mask.clone().detach()
        
        output = self.transformer(src_enc, tgt_enc,
                                  src_mask=self.masks['src'],
                                  tgt_mask=self.masks['tgt'],
                                  memory_mask=self.masks['memory'],
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  )
        output = self.out(output)
        return output