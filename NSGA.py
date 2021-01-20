# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:00:32 2019

@author: 1MLightyears

这是一个机器学习分组数据的算法，问题描述来自http://python.jobbole.com/82208/
由于网页中给出的代码在Spyder中编译报错，因此我决定用别的算法来完成这个例子。

本代码本质上是一个“从头造轮子”的内容。

代码逻辑：
step 1 生成数据（NSGA.makedata)
step 2 随机生成一批亲代(NSGA.train)
step 3 进化出下一代子代(NSGA.evolve)
       为保证亲代里的优秀个体不丢失，将亲代也加入子代(NSGA.train)
step 4 得到误差（NSGA.fit）
step 5 对子代依照误差值排序（NSGA.qsort），末尾淘汰(NSGA.train)
step 6 返回step 3直到代数限制(NSGA.train)

展示结果时，使用NSGA.py 中的show()方法，读取子代结果并展示
一组新数据的分组结果

由于希望对sklearn.datasets.make_moons()给出的数据进行分组，因此为分组结果建立数学模型：

f(x)=a0+a1*x+a2*x^2+a3*x^3

因此每代子代的基因型为[a0,a1,a2,a3]，对这个进行自然选择，选择方式为四性繁殖（取出4个优秀亲代
代并交叉配对产生子代），每代保留parentcount个优秀个体作为下一代的亲代，其余淘汰，如此多代
繁衍最终选择出优秀的子代。

后续可以考虑用数组保存a并且做一个启发式学习，可以自己增加多项式的阶数
"""
#%% import
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import time

#%% sample
class sample:
    a0,a1,a2,a3=0,0,0,0
    s=0
    def __init__(self):
        self.a0=np.random.uniform(-1,1)
        self.a1=np.random.uniform(-1,1)
        self.a2=np.random.uniform(-1,1)
        self.a3=np.random.uniform(-1,1)
        self.s=0
    def f(self,x):
        return self.a0+self.a1*x+self.a2*x**2+self.a3*x**3
    def fit(self):#计算子代匹配度
        '''这里我们希望，某一个子代能画出一条线尽可能的区分上下两组数据，集中在上面的是0，
        下面的是1，那么这就是一个很好的子代结果。
        每正确分组一个数据，这个子代的s就加1，那么s越高的子代自然就是越优秀的子代，选中
        他产生下一代的几率就越高。
        '''
        self.s=0
        global data,haty
        for i in range(len(haty)):
            if (haty[i]==0)and(self.f(data[i,0])>data[i,1]):
                self.s+=1
            if (haty[i]==1)and(self.f(data[i,0])<data[i,1]):
                self.s+=1
            #上面的红点是0，下面的蓝点是1
            #惩罚错误分类的数据点
        return self.s#越接近0的子代越好

#%% definition
parentcount=20     #每代保留下来的亲代个数
soncount=20        #产生的子代个数
parentgen=[]       #亲代
songen=[]          #子代
MutPer=0.1         #变异概率
EvoPa=0.2          #变异幅度
NumOfData=80       #训练用数据个数
epoch=20           #默认训练代数
Noise=0.15         #噪声
data=[]            #数据值
haty=[]            #真实值
bestson=sample()   #训练出来的最优子代
per=[]             #亲代交叉选择的权值表

#%% select
def select():#加权选出fit得分高的亲代，返回一个亲代的序号
    global per
    return np.random.choice(np.arange(len(per)),p=per/sum(per))
    #return 0
#%% evolve
def evolve():#四性繁殖出一个新子代
    global parentgen,per
    son=sample()
    #依照加权值进行交叉选择
    son.a0=parentgen[select()].a0#使用select函数是为了加权选择
    son.a1=parentgen[select()].a1
    son.a2=parentgen[select()].a2
    son.a3=parentgen[select()].a3
    if np.random.random()<MutPer:#发生变异
        son.a0*=1+np.random.uniform(-EvoPa,EvoPa)
        son.a1*=1+np.random.uniform(-EvoPa,EvoPa)
        son.a2*=1+np.random.uniform(-EvoPa,EvoPa)
        son.a3*=1+np.random.uniform(-EvoPa,EvoPa)
    '''while son.a0>0.5:
        son.a0/=2
    if son.a0<0:
        son.a0=-son.a0        '''
    return son

#%%前端
def makedata(N):#产生NumOfData个初始数据，存到data.npy和haty.npy里
    global data,haty,NumOfData
    NumOfData=N
    data, haty = sklearn.datasets.make_moons(NumOfData, shuffle=False, noise=Noise)
    #生成数据
    np.save('data.npy',data)
    np.save('haty.npy',haty)
    f=open('data.txt','w')
    for i in range(data.shape[0]):
        f.write(str('%4d'%(i))+':haty= '+str(haty[i])+', data= '+str(data[i,:])+'\n')
    f.close()

def plotdata():#对测试数据绘图
    global data,haty
    data=np.load('data.npy')
    haty=np.load('haty.npy')
    plt.scatter(data[:,0], data[:,1], s=2000//NumOfData, c=haty, cmap=plt.cm.Spectral)
    plt.show()
    #这里用一个除式决定s的值是为了一劳永逸地解决数据点堆在一起看不清的问题

def show():#向屏幕打印bestson
    #运行结束的main()会把结果输出到son.txt里，读取它
    global bestson
    f=open('son.txt','r')
    r=f.readline();bestson.a0=eval(r.split()[0])
    r=f.readline();bestson.a1=eval(r.split()[0])
    r=f.readline();bestson.a2=eval(r.split()[0])
    r=f.readline();bestson.a3=eval(r.split()[0])
    r=f.readline();bestson.s=eval(r.split()[0])
    print('最佳子代:')
    print('a0=%3.6f, a1=%3.6f, a2=%3.6f, a3=%3.6f'%(bestson.a0,bestson.a1,bestson.a2,bestson.a3))
    print('对于训练数据，最小损失函数值')
    print('s=%3.6f'%(bestson.s))
    f.close()

def test(x,y):#对于某个特定的数据，返回预测结果
    global bestson
    f=open('son.txt','r')
    r=f.readline();bestson.a0=eval(r.split()[0])
    r=f.readline();bestson.a1=eval(r.split()[0])
    r=f.readline();bestson.a2=eval(r.split()[0])
    r=f.readline();bestson.a3=eval(r.split()[0])
    r=f.readline();bestson.s=eval(r.split()[0])
    if bestson.f(x)<=y:
        print(0)
    else:
        print(1)
    f.close()

def ansplot():#画最佳结果的一个曲线图
    global bestson
    f=open('son.txt','r')
    r=f.readline();bestson.a0=eval(r.split()[0])
    r=f.readline();bestson.a1=eval(r.split()[0])
    r=f.readline();bestson.a2=eval(r.split()[0])
    r=f.readline();bestson.a3=eval(r.split()[0])
    r=f.readline();bestson.s=eval(r.split()[0])
    x=np.linspace(-1.5,2.5,201)
    y=bestson.f(x)
    plt.plot(x,y,color='green',linewidth=2)
    plt.ylim(-1.5,2)
    plt.show()

#%% qsort
def qsort(a,b):#对songen进行快速排序，以songen.s为比较标准，这里要从小到大排序
               #越接近0越好
    global songen
    mid=songen[b]
    i,j=a,b
    while i<j:
        while (i<j)and(songen[i].s<=mid.s):
            i+=1
        songen[i],songen[j]=songen[j],songen[i]
        while (i<j)and(songen[j].s>=mid.s):
            j-=1
        songen[i],songen[j]=songen[j],songen[i]
    songen[j]=mid
    if a<i:
        qsort(a,i-1)
    if j<b:
        qsort(j+1,b)
    return None
#%% train
def train(epoch):#进行训练，epoch=训练代数
    global data,haty,parentgen,songen,per
    data=np.load('data.npy')
    haty=np.load('haty.npy')
    haty=haty.tolist()
    #数据可视化准备
    plt.ion()
    x=np.linspace(-1.5,2.5,201)
    #首批亲代准备
    for i in range(parentcount):
        parentgen.append(sample())#随机生成一批亲代
        parentgen[i].fit()#赋值为得到每一个parentgen[i].s的值
    #设置EvoPa和MutPer的值
    #经过测试，建议EvoPa和MutPer大约设成 样本个数/代数 和 样本个数/代数/4 比较容易收敛
    EvoPa=NumOfData/epoch;MutPer=EvoPa/4
    #代繁殖循环
    for m in range(epoch):
        #计算亲代交叉选择的权值
        print('Gen %d, y= %.3f + %.3f*x + %.3f*x^2 + %.3f*x^3, fitness=%d'
              %(m,parentgen[0].a0,parentgen[0].a1,parentgen[0].a2,parentgen[0].a3,
                parentgen[0].s))
        per=np.ones(len(parentgen),dtype=int)#为防止除以0所以用ones而不是zeros
        for i in range(len(parentgen)):
            per[i]+=parentgen[i].s

        songen.clear();songen=[]
        for i in range(soncount):
            songen.append(evolve())#亲代交叉选择繁衍子代，存入songen
        songen+=parentgen#为不丢失可能更加优秀的亲代个体，把亲代也放到下一代子代去
        for i in range(soncount+parentcount):#子代fit得到各子代的s值
            songen[i].fit()
        qsort(0,soncount+parentcount-1)#子代排序
        parentgen.clear();parentgen=songen[0:parentcount]#选出优秀的parentcount
                                                         #个子代成为新亲代
        #可视化部分
        y=parentgen[0].f(x)
        ax=plt.gca()
        ax.remove()
        plt.ylim(-1.5,2)
        plt.scatter(data[:,0], data[:,1], s=2000//NumOfData, c=haty, cmap=plt.cm.Spectral)
        plt.plot(x,y,color='green',linewidth=1,alpha=0.5)
        plt.pause(0.05)
    print('EvoPa=%.3f,MutPer=%.3f, 准确率%.3f%%'%(EvoPa,MutPer,100-parentgen[0].s/NumOfData//0.00001/1000))
    plt.ioff();plt.show()
    #自然选择结束，现在认为最优秀的子代在songen[0],把他保存到son.txt中
    f=open('son.txt','w')
    f.write(str(songen[0].a0)+'\n')
    f.write(str(songen[0].a1)+'\n')
    f.write(str(songen[0].a2)+'\n')
    f.write(str(songen[0].a3)+'\n')
    f.write(str(songen[0].s)+'\n')
    f.close()
    return None

#%% main
if __name__=='__main__':
    np.random.seed(int(time.time()))
    s=''
    cmdlist=[]
    while s!='exit':
        s=input('>')
        cmdlist=s.split()
        if cmdlist==[]:comlist=['None']             #防止只敲一个回车
        if cmdlist[0]=='makedata':                  #生成数据，默认NumOfData个
            if len(cmdlist)>1:
                makedata(int(cmdlist[1]))
            else:
                makedata(NumOfData)

        elif cmdlist[0]=='plotdata':                #数据作图
            plotdata()

        elif cmdlist[0]=='show':
            show()

        elif cmdlist[0]=='train':                   #训练，默认epoch代
            if len(cmdlist)>1:
                train(int(cmdlist[1]))
            else:
                train(epoch)
        elif cmdlist[0]=='noise':                   #生成数据的噪声，默认0.1
            if len(cmdlist)>1:
                Noise=float(cmdlist[1])
            else:
                print('当前noise=',Noise,sep='')
        elif cmdlist[0]=='test':                   #测试对某一对数据的结果
            if len(cmdlist)>2:
                test(eval(cmdlist[1]),eval(cmdlist[2]))
            else:
                print('输入错误，格式: test x y')
        elif cmdlist[0]=='ansplot':                #打印一个结果
            ansplot()
        elif cmdlist[0]=='print':                  #输出numpy格式的data数据
            data=np.load('data.npy')
            haty=np.load('haty.npy')
            haty=haty.tolist()
            print(data)
            print(haty)
        elif cmdlist[0]=='exit':
            pass
        else:
            print('输入错误\n可用命令:\n makedata, plotdata, show, train, noise, test, ansplot, print')




