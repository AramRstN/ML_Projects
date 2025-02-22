import Transformer as tf
import torch.nn as nn
import torch.nn.functional as f

#cross-attention
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super(TransformerDecoder, self).__init__()
        self.embedding = tf.InputEmbedding(vocab_size, d_model)
        self.positional_embedding = tf.PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([tf.DecoderLayer(d_model,num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, tgt_mask, cross_mask):
        x = self.embedding(x)
        x = self.positional_embedding(x)
        for layer in self.layers:
            x = layer(x, y, tgt_mask, cross_mask)
        x = self.fc(x)
        return f.log_softmax(x, dim=-1)
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = tf.MultiHeadAttention(d_model, num_heads)
        self.cross_attn = tf.MultiHeadAttention(d_model, num_heads)
        self.ff_sublayer = tf.FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, tgt_mask, cross_mask):
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        cross_attn_output = self.cross_attn(x, y, y, cross_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.ff_sublayer(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
class Transformer (nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super().__init__()

        self.encoder = tf.TransformerEncoder(vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_seq_length)
        self.decoder = tf.TransformerDecoder(vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_seq_length)

    def forward(self, x, src_mask, tgt_mask, cross_mask):
        encoder_output = self.encoder(x, src_mask)
        decoder_output = self.decoder(x, encoder_output, tgt_mask, cross_mask)
        return decoder_output