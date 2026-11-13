import torch
import torch.nn as nn

vocab_size = 20
embedding_dim = 8
seq_len = 5
batch_size = 1

input_ids = torch.tensor([
    [2,3,5,6,1],
    [7,9,2,3,5],
    [9,0,2,5,1]
])

token_embd = nn.Embedding(vocab_size,embedding_dim)
token_vectors = token_embd(input_ids)

position_ids = torch.arange(seq_len)
position_embd = nn.Embedding(seq_len,embedding_dim)
position_vectors = position_embd(position_ids)

total_vectors = token_vectors + position_vectors

print(token_vectors.shape)
print(position_vectors.shape)
print(total_vectors.shape)