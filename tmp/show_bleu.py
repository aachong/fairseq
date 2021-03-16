import plotly.graph_objs as go

x,y=[],[]
while True:
    try:
        ls = input().split(" ")
        x.append(int(ls[ls.index('epoch')+1]))
        y.append(int(ls[ls.index('bleu')+1]))
        print(x,y)
    except Exception as e:
        print(e)
        break

# x = ['23','epoch']
# print(x.index('epoch'))
# input()
# line1 = go.Scatter(x=,y=,name='')
# line2 = go.Scatter(x=,y=,name='')
# fig = go.Figure([line1,line2])
# fig.update_layout(title='',xaxis_title='',yaxis_title='')
# fig.show()