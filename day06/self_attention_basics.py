import torch
import torch.nn.functional as F

Q = torch.tensor([
    [1.0,0.1,0.0],
    [0.0,1.0,1.0]
])

K = torch.tensor([
    [0.0,1.0,0.0],
    [0.1,1.0,0.0]
])

V = torch.tensor([
    [0.0,1.0,1.1],
    [10.0,0.1,5.0]
])

scores = Q @ K.T
scores = scores/(K.shape[-1]**0.5)
weights = F.softmax(scores,dim=-1)
output = weights @ V
print(output)
print(output.shape)