import torch
import torch.nn as nn

vocab_size = 10
embedding_dim = 4
block_size = 4

input_ids = torch.tensor([
    [2,5,1,7],
    [3,4,6,8]
])

token_embd = nn.Embedding(vocab_size,embedding_dim)
token_vectors = token_embd(input_ids)

position_ids = torch.arange(block_size)
position_embd = nn.Embedding(block_size,embedding_dim)
position_vectors = position_embd(position_ids)

input_vectors = token_vectors + position_vectors

print("Token Embeddings : ",token_vectors.shape)
print("Position embeddings:", position_vectors.shape)
print("Final input embeddings:", input_vectors.shape)

print(token_vectors)
print(position_vectors)
print(input_vectors)