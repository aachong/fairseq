import torch
import torch.nn as nn
import pickle
import numpy as np
from scipy import stats
#判断噪音是否服从正太分布，但是不符合，现在还不知道服从什么分布

def get_data(file):
    with open('./tmp/'+file+'.out','rb') as f:
        data:torch.tensor = pickle.load(f)
    return data

def save_data(data,file):
    with open('./tmp/'+file+'.out','wb') as f:
        pickle.dump(data,f)

d = get_data('datax')
d = d.reshape(-1)
save_data(d,'datax')
input_logits=torch.randn(10,20,30)
inp = input_logits.detach()
data=d
idx = (torch.rand(inp.shape)*(len(data)-10)).long()
noise = data[idx]
x = torch.randn(30)
y = torch.tensor([[1,2],[1,4]])
x[y]
x
def em(data):
    e = data.std()
    m = data.mean()
    print(e,m)
    return e,m
x=torch.rand(10000)*8704950
x = torch.randn(34)
x.int
for i in range(0,10):
    try:
        data = get_data('data_'+str(i))
        data = data.reshape(-1)
        e,m = em(data)
    except:
        print('no')
# x,y=[],[]
# for i in data:
#     a,b=em(i.reshape(-1))
#     x.append(float(a))
#     y.append(float(b))

# print(x,y)
# data = data[0].reshape(-1)

# data = data.numpy()
# size = data.shape
# print(size)

# e,m = em(data)
x = stats.kstest(data, 't', (m, e))
print(x)
# data = np.random.normal(m,e,size)
# em(data)
# import plotly.graph_objs as go
# line1 = go.Histogram(x=data,xbins={'size':2},name='data') #添加统计间隔，每隔都少个加一个
# fig = go.Figure(line1)
# fig.update_layout(title='ex',xaxis_title='x',yaxis_title='y')  #添加间隙
# fig.show()

import numpy as np

x = np.random.binomial(19000,0.3,(19000,))
y = np.random.binomial(19000,0.3,(19000,))
(x-y).std()
y.std()

a = np.random.normal(0,0.3,(19000,))
b = np.random.normal(0,0.3,(19000,))
a.std()
(a-b).std()