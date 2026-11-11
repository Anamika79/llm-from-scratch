import torch
import torch.nn as nn

token_embedding = nn.Embedding(4, 3)
input_ids = torch.tensor([0, 1, 2])
token_vectors = token_embedding(input_ids)

print(token_vectors)
print(token_vectors.shape)

position_embedding = nn.Embedding(3,3)
position_ids = torch.arange(3)
position_vectors = position_embedding(position_ids)

final_embedding = token_vectors + position_vectors
print(final_embedding)
print(final_embedding.shape)