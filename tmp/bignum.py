class Solution:
    ###因为本实验主要实现大数乘法，所以大数加法、大数减法、
    ###字符串向int类型转换、int类型向字符串类型转换，这四个方法
    ###采用python内置的函数。如下

    #大数减法
    def sub(self,a,b):
        return str(int(a)-int(b))

    #大数加法
    def add(self,*arg):
        ans = 0
        for i in arg:
            ans += int(i)
        return str(ans)

    #字符串向int转换
    # int = int(str)

    #int向字符串转换
    # str = str(int)

    ############################
    ####接下来为分治法2求解大数乘法
    
    def multiply(self, a: str, b: str) -> str:
            #当a和b的长度都小于4的时候，那么int32位可以存下，所以直接转化为int类型计算输出
        if len(a)<=4 and len(b)<=4: 
            return str(int(a)*int(b))

        #判断前边有没有负号，False没有负号，True有负号    
        prefix = False   

        # 判断最后的结果之前应不应该加符号
        if a[0]=='-':
            a = a[1:]
            prefix = not prefix
        if b[0]=='-':
            b = b[1:]
            prefix = not prefix

        # 分治法要两个数长度相等，若一个长度比另一个小，则在短的前边补0
        if not len(a) == len(b):
            if len(a)>len(b):
                b = '0'*(len(a)-len(b)) + b
            else: a = '0'*(len(b)-len(a)) + a
        # 验证两个数长度相等
        assert len(a) == len(b)
        na = len(a)
        nb = len(b)

        #把数a分成a1，a2
        a1 = a[:na//2]
        a2 = a[na//2:na]

        #把数b分成b1，b2
        b1 = b[:nb//2]
        b2 = b[nb//2:nb]

        #计算分治中的三次乘法
        a1b1 = self.multiply(a1,b1)
        a2b2 = self.multiply(a2,b2)
        aabb = self.add(self.multiply(self.sub(a1,a2),self.sub(b2,b1)), a1b1 , a2b2)

        #把三次乘法算出来的结果跟据公式整合起来
        ans = self.add(a1b1+'0'*(len(a2)+len(b2)),aabb+'0'*len(a2),a2b2)

        #如果前边有负号则添加负号
        ans = ('-' if prefix else '') + ans
        return str(ans)

def test():
    a = Solution()
    print(a.multiply('12312442','324234324'))
test()