import plotly.graph_objs as go

x,y=[],[]
while True:
    try:
        ls = input().split(" ")
        x.append(int(ls[ls.index('epoch')+1]))
        y.append(float(ls[ls.index('bleu')+1]))
    except Exception as e:
        break
print(x,y)
line1 = go.Scatter(x=x,y=y,name='bleu')
fig = go.Figure([line1])
fig.update_layout(title='bleu',xaxis_title='epoch',yaxis_title='bleu')
fig.show()