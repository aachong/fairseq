s = ''
with open('/home/rcduan/fairseq/fairseq/examples/_transformer_base/data-bin/code.ch','r') as f:
    x=1
    print(32)
    for i in f.readlines():
        s +=' '.join(i.split(' ')[:2])+'\n'

with open('/home/rcduan/fairseq/fairseq/examples/_transformer_base/data-bin/code.ch','w') as f:
    f.write(s)