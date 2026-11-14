import torch

# block_size = 4
# mask = torch.tril(torch.ones(block_size,block_size))
# print(mask)

scores = torch.tensor([
    [2., 1., 3., 4.],
    [1., 3., 2., 5.],
    [2., 4., 3., 1.],
    [3., 2., 1., 4.]
])

block_size = 4
mask = torch.tril(torch.ones(block_size,block_size))
masked_scores = scores.masked_fill(mask==0,float('-inf'))
print(masked_scores)

weights = torch.softmax(masked_scores,dim=-1)
print(weights)