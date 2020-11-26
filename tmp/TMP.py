import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

a = torch.randn(3,4,5).cuda()
print(a)
x = a.topk(1,-1)
print(x)
a.scatter_add_(-1,x[1],torch.ones_like(x[1])*0.5)
print(a)