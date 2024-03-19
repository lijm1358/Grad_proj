import math
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertForMaskedLM

class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_units, dropout_prob):
        super(ScaledDotProductAttention, self).__init__()
        self.head_units = head_units
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, Q, K, V, mask):
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_units)
        attn_score = attn_score.masked_fill(mask == 0, -1e9)
        attn_dist = self.dropout(F.softmax(attn_score, dim=-1))
        # dim of output : batchSize x num_head x seqLen x head_units
        output = torch.matmul(attn_dist, V)
        return output, attn_dist


class MultiHeadAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dropout_prob):
        super(MultiHeadAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.head_units = self.hidden_size // self.num_attention_heads

        # query, key, value, output
        self.W_Q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_K = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_O = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.attention = ScaledDotProductAttention(self.head_units, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.layerNorm = nn.LayerNorm(self.hidden_size, 1e-6)

    def forward(self, enc, mask):
        residual = enc  # residual connection
        batch_size, seqlen = enc.size(0), enc.size(1)

        Q = self.W_Q(enc).view(batch_size, seqlen, self.num_attention_heads, self.head_units)
        K = self.W_K(enc).view(batch_size, seqlen, self.num_attention_heads, self.head_units)
        V = self.W_V(enc).view(batch_size, seqlen, self.num_attention_heads, self.head_units)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        output, attn_dist = self.attention(Q, K, V, mask)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seqlen, -1)

        output = self.layerNorm(self.dropout(self.W_O(output)) + residual)
        return output, attn_dist

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dropout_prob):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.head_units = self.hidden_size // self.num_attention_heads

        # query, key, value, output
        self.W_Q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_K = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_O = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.attention = ScaledDotProductAttention(self.head_units, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.layerNorm = nn.LayerNorm(self.hidden_size, 1e-6)

    def forward(self, enc, gen, mask):
        residual = gen  # residual connection
        batch_size, seqlen = enc.size(0), enc.size(1)

        Q = self.W_Q(gen).view(batch_size, seqlen, self.num_attention_heads, self.head_units)
        K = self.W_K(enc).view(batch_size, seqlen, self.num_attention_heads, self.head_units)
        V = self.W_V(enc).view(batch_size, seqlen, self.num_attention_heads, self.head_units)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        output, attn_dist = self.attention(Q, K, V, mask)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seqlen, -1)

        output = self.layerNorm(self.dropout(self.W_O(output)) + residual)
        return output, attn_dist

class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(PositionwiseFeedForward, self).__init__()

        self.W_1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.W_2 = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layerNorm = nn.LayerNorm(hidden_size, 1e-6)

    def forward(self, x):
        residual = x
        output = self.W_2(F.gelu(self.dropout(self.W_1(x))))
        output = self.layerNorm(self.dropout(output) + residual)
        return output
    
    
class BERT4RecEncBlock(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dropout_prob):
        super(BERT4RecEncBlock, self).__init__()
        self.attention = MultiHeadAttention(num_attention_heads, hidden_size, dropout_prob)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_size, dropout_prob)

    def forward(self, input_enc, mask):
        output_enc, attn_dist = self.attention(input_enc, mask)
        output_enc = self.pointwise_feedforward(output_enc)
        return output_enc, attn_dist
    
class BERT4RecDecBlock(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dropout_prob):
        super(BERT4RecDecBlock, self).__init__()
        self.attention = MultiHeadAttention(num_attention_heads, hidden_size, dropout_prob)
        self.cross_attention = MultiHeadCrossAttention(num_attention_heads, hidden_size, dropout_prob)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_size, dropout_prob)

    def forward(self, input_enc,input_gen, mask):
        output_enc, attn_dist = self.attention(input_enc, mask)
        output_dec, attn_dist = self.cross_attention(output_enc, input_gen, mask)
        output_dec = self.pointwise_feedforward(output_dec)
        return output_dec, attn_dist
    

class CrossBERT4Rec(nn.Module):
    def __init__(
        self,
        num_item,
        gen_img_emb,
        hidden_size=256,
        num_attention_heads=4,
        num_enc_hidden_layers=3,
        num_dec_hidden_layers=3,
        hidden_act="gelu",
        num_gen_img=1,
        max_len=30,
        dropout_prob=0.2,
        pos_emb=False,
        num_mlp_layers=2,
        device="cpu",
    ):
        super(CrossBERT4Rec, self).__init__()

        self.num_item = num_item
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_enc_hidden_layers = num_enc_hidden_layers
        self.num_dec_hidden_layers = num_dec_hidden_layers
        self.device = device
        self.pos_emb = pos_emb
        self.num_mlp_layers = num_mlp_layers
        self.num_gen_img = num_gen_img
        self.gen_img_emb = gen_img_emb.to(self.device)  # (num_item) X (3*512)

        self.item_emb = nn.Embedding(num_item + 2, hidden_size, padding_idx=0)
        #self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.emb_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        
        self.block_layers = [BERT4RecEncBlock(num_attention_heads,hidden_size, dropout_prob) for _ in range(num_enc_hidden_layers)]        
        self.block_layers += [BERT4RecDecBlock(num_attention_heads,hidden_size, dropout_prob) for _ in range(num_dec_hidden_layers)]

        self.cross_bert = nn.ModuleList(self.block_layers)
        ###여기부터
        # init MLP
        
        self.out = nn.Linear(self.hidden_size, self.num_item + 1)

    def forward(self, log_seqs, labels):
        seqs = self.item_emb(log_seqs).to(self.device)
        
        if self.pos_emb:
            positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
            seqs += self.pos_emb(torch.tensor(positions).to(self.device))
        seqs = self.emb_layernorm(self.dropout(seqs))

        mask = (log_seqs > 0).unsqueeze(1).repeat(1, log_seqs.shape[1], 1).unsqueeze(1).to(self.device)

        item_ids = log_seqs.clone().detach()
        mask_index = torch.where(item_ids == self.num_item + 1)  # mask 찾기
        item_ids[mask_index] = labels[mask_index]  # mask의 본래 아이템 번호 찾기
        # log_seq는 원래 아이템 id + 1 되어 있으므로 -1해서 인덱싱
        img_idx = sample([0, 1, 2], k=self.num_gen_img)  # 생성형 이미지 추출
        gen_imgs = torch.flatten(self.gen_img_emb[item_ids - 1][:, :, img_idx, :], start_dim=-2, end_dim=-1)

        for i in range(len(self.cross_bert)):
            if self.num_enc_hidden_layers > i:
                seqs, _ = self.cross_bert[i](seqs,mask)
            else:
                seqs, _ = self.cross_bert[i](seqs,gen_imgs,mask)

        out = self.out(seqs)

        return out