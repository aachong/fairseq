import torch
import torch.nn as nn
import plotly.graph_objs as go
import plotly
import plotly.io as pio`

x = torch.randn(5,1000,100)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        print(self.a_2.shape)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

m = nn.LayerNorm(x.shape[1:])
x = m(x)
x.detach_()
trace0 = go.Histogram(x=x[0].reshape(-1))
data = [trace0]
fig_spx = go.Figure(data=data)
# fig_spx.show()


