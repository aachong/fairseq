数据采用从ldc中随机抽样了20w句话

模型依然是标准的transformer

对比的是cross_entropy和r3f这两种loss

对比了closer_all和r3f

前者比后者有一个点的提升，并且收敛速度比后这快

学习率应该调大，lambda还可以再往大调调，前几层之前的参数也可以再往大调调

说明每一层拉近距离还是很有必要的