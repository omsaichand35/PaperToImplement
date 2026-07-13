import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.encoder_stack = 6
        self.decoder_stack = 6
        self.dmodel = 512
        self.dk = 64
        self.dv = 64
        self.h = 8
        self.warmup_steps = 10000

        # [Paper Spec Section 3.3] Encoder-Decoder Structure
        self.encoder = Encoder(self.encoder_stack, self.dmodel, self.dk, self.dv, self.h)
        self.decoder = Decoder(self.decoder_stack, self.dmodel, self.dk, self.dv, self.h)

        # [Paper Spec Section 3.3] Embeddings and Softmax
        self.embedding = nn.Embedding(self.dmodel, self.dmodel)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_seq):
        # [Paper Spec Section 3.3] Encoder-Decoder Structure
        encoder_output = self.encoder(input_seq)
        decoder_output = self.decoder(encoder_output)

        # [Paper Spec Section 3.3] Embeddings and Softmax
        embedding_output = self.embedding(decoder_output)
        softmax_output = self.softmax(embedding_output)

        return softmax_output


class Encoder(nn.Module):
    def __init__(self, num_layers, dmodel, dk, dv, h):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.dmodel = dmodel
        self.dk = dk
        self.dv = dv
        self.h = h

        # [Paper Spec Section 3.3] Scaled Dot-Product Attention
        self.self_attn = MultiHeadAttention(self.num_layers, self.dmodel, self.dk, self.dv, self.h)

        # [Paper Spec Section 3.3] Position-wise Feed-Forward Networks
        self.feed_forward = PositionWiseFeedForward(self.dmodel, self.dk, self.dv, self.h)

    def forward(self, input_seq):
        # [Paper Spec Section 3.3] Scaled Dot-Product Attention
        self_attn_output = self.self_attn(input_seq, input_seq, input_seq)

        # [Paper Spec Section 3.3] Position-wise Feed-Forward Networks
        feed_forward_output = self.feed_forward(self_attn_output)

        return feed_forward_output


class Decoder(nn.Module):
    def __init__(self, num_layers, dmodel, dk, dv, h):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.dmodel = dmodel
        self.dk = dk
        self.dv = dv
        self.h = h

        # [Paper Spec Section 3.3] Scaled Dot-Product Attention
        self.self_attn = MultiHeadAttention(self.num_layers, self.dmodel, self.dk, self.dv, self.h)

        # [Paper Spec Section 3.3] Position-wise Feed-Forward Networks
        self.feed_forward = PositionWiseFeedForward(self.dmodel, self.dk, self.dv, self.h)

    def forward(self, input_seq):
        # [Paper Spec Section 3.3] Scaled Dot-Product Attention
        self_attn_output = self.self_attn(input_seq, input_seq, input_seq)

        # [Paper Spec Section 3.3] Position-wise Feed-Forward Networks
        feed_forward_output = self.feed_forward(self_attn_output)

        return feed_forward_output


class MultiHeadAttention(nn.Module):
    def __init__(self, num_layers, dmodel, dk, dv, h):
        super(MultiHeadAttention, self).__init__()
        self.num_layers = num_layers
        self.dmodel = dmodel
        self.dk = dk
        self.dv = dv
        self.h = h

        # [Paper Spec Section 3.3] Scaled Dot-Product Attention
        self.query_linear = nn.Linear(self.dmodel, self.dk)
        self.key_linear = nn.Linear(self.dmodel, self.dk)
        self.value_linear = nn.Linear(self.dmodel, self.dv)

    def forward(self, query, key, value):
        # [Paper Spec Section 3.3] Scaled Dot-Product Attention
        query_linear_output = self.query_linear(query)
        key_linear_output = self.key_linear(key)
        value_linear_output = self.value_linear(value)

        # [Paper Spec Section 3.3] Scaled Dot-Product Attention
        attention_weights = torch.matmul(query_linear_output, key_linear_output.T) / math.sqrt(self.dk)
        attention_output = torch.matmul(attention_weights, value_linear_output)

        return attention_output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, dmodel, dk, dv, h):
        super(PositionWiseFeedForward, self).__init__()
        self.dmodel = dmodel
        self.dk = dk
        self.dv = dv
        self.h = h

        # [Paper Spec Section 3.3] Position-wise Feed-Forward Networks
        self.linear1 = nn.Linear(self.dmodel, self.dk)
        self.linear2 = nn.Linear(self.dk, self.dv)

    def forward(self, input_seq):
        # [Paper Spec Section 3.3] Position-wise Feed-Forward Networks
        linear1_output = F.relu(self.linear1(input_seq))
        linear2_output = self.linear2(linear1_output)

        return linear2_output
