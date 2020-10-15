import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x = torch.tensor([1,2,3]).float()
print(x)
x = x.norm(2,0,keepdim=True)

print(x)