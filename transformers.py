import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer
import torch.nn.functional as F

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class BBoxPositionalEncoding(nn.Module):
    """
    Encodes the BBox centroid and page as one hot encoded vector.
    They are fed to fully connected layers.
    Responses are added together to obtain a resulting single tensor
    d_hid : hidden dimensionality of transformer
    positional_grid_shape : grid to use to discretize positions of bboxes
    max_pages: max number of pages in documents
    """
    def __init__(self,d_hid,positional_grid_shape,max_pages):
        super().__init__()
        self.d_hid = d_hid
        self.fc_x = nn.Linear(positional_grid_shape[1],d_hid)
        self.fc_y = nn.Linear(positional_grid_shape[0],d_hid)
        self.fc_page = nn.Linear(max_pages,d_hid)
    def forward(self,x,x_in,y_in,page_in):
        out = x
        for module,fv in zip([self.fc_x,self.fc_y,self.fc_page],
                             [x_in,y_in,page_in]):
            # o = F.relu(module(fv))
            out += module(fv)
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x,src_positions_x,src_positions_y,src_positions_page):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, positional_encoding,dropout=0.1, n_position=200):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = positional_encoding#PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask, src_positions_x,src_positions_y,src_positions_page,return_attns=False):

        enc_slf_attn_list = []

        # -- Forward

        enc_output = self.dropout(self.position_enc(self.src_word_emb(src_seq),src_positions_x,src_positions_y,src_positions_page))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class TransformerTextClassifier(nn.Module):
    """
    Tranformer adaptation for text classification.
    Positional encoding: "sequence" or "bbox"
    positional_grid_shape : grid to use to discretize positions of bboxes
    max_pages: max number of pages in documents

    """

    def __init__(
            self, n_src_vocab, src_pad_idx,
            d_word_vec=256, d_model=256, d_inner=512,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            n_position=200,positional_encoding='sequence',positional_grid_shape=(1245,862),max_pages=8):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        if positional_encoding == 'sequence':
            positional_encoder = PositionalEncoding(d_word_vec, n_position=n_position)
        elif positional_encoding == 'bbox':
            positional_encoder = BBoxPositionalEncoding(d_word_vec,positional_grid_shape=positional_grid_shape,max_pages=max_pages)#, positional_grid_shape=bbox_enc_shape,max_pages=max_pages)
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, positional_encoding=positional_encoder,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout)

        self.fc = nn.Linear(d_model,1)
        # self.last_conv = nn.Conv1d(,1,d_model)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        self.x_logit_scale = 1.


    def forward(self, src_seq,src_positions_x,src_positions_y,src_positions_page):

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)

        enc_output,enc_slf_attn_list  = self.encoder(src_seq, src_mask,src_positions_x,src_positions_y,src_positions_page,return_attns=True)
        # enc_output = enc_output.sum(axis=-1)
        # enc_output = enc_output.transpose(1,2)
        # out = self.last_conv(enc_output).view(src_seq.shape[0],-1)
        out = self.fc(enc_output).view(src_seq.shape[0],-1)

        return out
