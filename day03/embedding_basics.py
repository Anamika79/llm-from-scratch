import torch
import torch.nn as nn

vocabulary = ['a', 'b', 'c', 'd']

char_to_id = {}
for index,char in enumerate(vocabulary):
    char_to_id[char] = index

print("Character to id : ", char_to_id)

embedding = nn.Embedding(4,3)
print(embedding.weight)

input_ids = torch.tensor([2, 0, 1])

output = embedding(input_ids)
print(output)
print(output.shape)

print(embedding(torch.tensor([0])))
print(embedding(torch.tensor([0,1])))
print(embedding(torch.tensor([0,1,2,3])))

print(embedding.weight.requires_grad)