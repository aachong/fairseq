import torch
import torch.nn as nn
import torch.nn.functional as F

a1 = torch.randn(10,11,requires_grad=True)
x = a1.chunk(10,0)
a,b = x[0],x[1]
b1 = a+b
c = b1*2
c.backward()
a.grad
b.grad
a1.grad

a3 = a1+a2
a3
c = a3*2
c.backward()
a2.grad
a1.grad