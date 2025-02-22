import torch
import math
import torch.nn as nn
import torch.nn.functional as f

model = nn.Transformer(
    d_model= 1536,
    nhead= 8,
    num_encoder_layers= 6,
    num_decoder_layers= 6
)

class InputEmbedding (nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model,max_seq_length):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype= torch.float) * -(math.log(10000.0)/d_model))

        pe[:,0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer ('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
embedding_layer = InputEmbedding(vocab_size= 10000, d_model=512)
# output = embedding_layer(torch.tensor([[1,2,3,4], [5,6,7,8]]))
# print(output.shape)
pos_encoding_layer = PositionalEncoding(d_model=512, max_seq_length=4)
# output = pos_encoding_layer(token_embeddings)
# print(output.shape)
# print(output[0][0][:10])

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.query_Linear = nn.Linear(d_model, d_model, bias=False)
        self.value_Linear = nn.Linear(d_model, d_model, bias=False)
        self.key_Linear = nn.Linear(d_model, d_model, bias=False)

        self.output_Linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        seq_length = x.size(1)
        x = x.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
    
    def compute_attention(self, q, k, v, mask= None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = f.softmax(scores, dim= -1)
        return torch.matmul(attention_weights, v)
    
    def combine_heads(self, x, batch_size):
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, -1, self.d_model)
    

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.split_heads(self.query_Linear(q), batch_size)
        k = self.split_heads(self.key_Linear(k), batch_size)
        v = self.split_heads(self.value_Linear(v), batch_size)

        attention_weights = self.compute_attention(q, k, v, mask)

        output = self.combine_heads(attention_weights, batch_size)

        return self.output_Linear(output)

d_model = 512
num_heads = 8
query = []
key = []
value = []
multihead_attention = MultiHeadAttention(d_model, num_heads)
output = multihead_attention(query, key, value)


# create Transformer with above components
class FeedForwardSubLayer (nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff_sublayer = FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        attn_output = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff_sublayer(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super().__init__()
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, src_mask):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        logist = self.fc(x)
        return f.log_softmax(logist, dim = 1)
    
class RegressionHead(nn.Module):
    def __init__(self, d_model, output_dim):
        super().__init__()
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        return self.fc(x)



d_model= 512 
d_ff = 2048
input = []
feed_forward = FeedForwardSubLayer(d_model, d_ff)
output = feed_forward(input)

vocab_size = 1024
num_classes = 7
num_layers = 8
dropout = 0.5
seq_length = 6
input_sequence = []
src_mask = 0
batch_size = 2

encoder = TransformerEncoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, seq_length)
classifier = ClassificationHead(d_model, num_classes)
output = encoder(input_sequence, src_mask)
classification = classifier(output)
print(f"Classification outputs for a batch of 
      {batch_size} sequences:\n{classification}")
print(f"Encoder output shape: {output.shape}\n
      Classification head output shape: {classification.shape}")

tgt_mask = (1 - torch.triu(
    torch.ones(1, seq_length, seq_length), diagonal=1)
    ).bool()

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff_sublayer = FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff_sublayer(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super(TransformerDecoder, self).__init__()
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([DecoderLayer(d_model,num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, tgt_mask):
        x = self.embedding(x)
        x = self.positional_embedding(x)
        for layer in self.layers:
            x = layer(x, tgt_mask)
        x = self.fc(x)
        return f.log_softmax(x, dim=-1)

# Instantiate a decoder transformer and apply it to input_tokens and tgt_mask
max_seq_length = 10
input_tokens = []
transformer_decoder = TransformerDecoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length)   
output = transformer_decoder(input_tokens, tgt_mask)
print(output)
print(output.shape)