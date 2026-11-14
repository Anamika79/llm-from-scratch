import torch
import torch.nn.functional as F

#query,key,value
Q = torch.tensor([
    [1.0,0.0],
    [0.0,1.0],
    [1.0,1.0]
])

K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

V = torch.tensor([
    [10.0, 0.0],
    [0.0, 10.0],
    [5.0, 5.0]
])

#Caluclate attention score. @ performs matrix multiplication, .T transposes the Key matrix.
scores = Q @ K.T
#Scale attention score . K.shape[-1] gives the Key dimension.
scores = scores/(K.shape[-1] ** 0.5)
#Calculate attention weights. dim=-1 applies Softmax across the last dimension.
weights = F.softmax(scores,dim=-1) 
#Final output
output = weights @ V

print("Attention Score : ",scores)
print("Attention weights : ",weights)
print("Final output : ",output)
print(output.shape)