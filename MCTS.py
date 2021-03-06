# 用 Monte Carlo Tree Search解决 寻路问题
# by 1mlightyears@gmail.com
# 20210215

import numpy as np
import matplotlib.pyplot as plt
from fire import Fire
from time import time
from copy import deepcopy
import seaborn as se
import sys

__all__=["requirements","node","MCTS","demo"]


requirements="""fire==0.4.0
matplotlib=3.3.3
numpy==1.19.4
seaborn==0.11.0
"""

class node:
    def __init__(self,
        x: int = 0,
        y: int = 0,
        weight: float = 1,
        level: int = 0
    ):
        """
        trails中的节点的数据结构。
        """
        self.x, self.y, self.weight, self.level = x, y, weight, level

    def __gt__(self, i):
        return self.level > i.level or ((self.level == i.level) and (self.weight > i.weight))

class MCTS:
    def __init__(self):
        """
        MCTS的演示对象。
        """
        self.threshold = 4
        self.factor = 4
        self.base = 30

        self.X, self.Y = [], []
        self.is_cli = False

    def __frame(self):
        """
        可视化部分:每帧刷新
        """
        for i in range(self.n):
            for j in range(self.m):
                have = False
                for node in self.trails:
                    if (i == node.x) and (j == node.y):
                        have = True
                        break
                self.C[i][j] = 2 if have\
                    else 3 if self.visited[i][j] else 5
        for i,j in self.barriers:
            self.C[i][j] = 4
        self.C[self.f[0]][self.f[1]] = 0
        self.C[self.t[0]][self.t[1]] = 1

    def __Ins(self, x, y, fr: node = node()):
        """
        将新试探点插入trails。
        x(int),y(int):新试探点的坐标；
        fr(node):新试探点由哪个点发展而来。
        新试探点的具体权重由旧点和它到终点的期望决定。
        """
        if (x, y) == self.t:
            return True
        sign = (abs(self.t[0] - fr.x) + abs(self.t[1] - fr.y)) - \
            (abs(self.t[0] - x) + abs(self.t[1] - y))

        # 权重函数
        # 1. 离终点越近的节点权重越高，离起点越近的函数权值越低
        # 2. 距终点距离相当的节点权重相当，相邻节点需要有明显差距
        # 3. 为防止权重爆炸，使用 p=weight*base^level 的科学计数法模式记录权值，显然weight<base
        # 4. 利用3.，每次只将level最大的那些节点的weight加入权重候选，除非节点数少于threshold个/level==0
        # 5. log(base,p)=ln p/ln base

        ln_weight = np.log(fr.weight)

        level = fr.level
        weight = fr.weight * self.factor ** sign
        if weight>self.base:
            level += 1
            weight /= self.base
        if weight < 1:
            level -= 1
            weight *= self.base

        self.trails.append(node(x, y, weight, level))
        self.visited[x][y] = True
        return False

    def setMap(self,
               n: int = 15,
               m: int = 15,
               f: tuple = (0, 0),
               t: tuple = (14, 14),
               b: list = [],
               sleep: float = 0.1,
               nograph: bool = False):
        """
        地图声明部分。
        n,m(int):地图长宽。
        f,t(tuple[int,int]):起点与终点。
        b(list[tuple[int,int]]):地图中不可通行的障碍物。
        sleep(float):控制试探间隔时间。
        nograph(bool):不显示可视化窗口。
        """

        self.visited = np.array([[False for j in range(m)] for i in range(n)])
        self.n = n
        self.m = m
        self.f = f
        self.t = t
        self.sleep = sleep
        self.nograph = nograph
        for i in b:
            self.visited[i[0]][i[1]] = True
        self.barriers = b
        seed = int(time())
        np.random.seed(seed)

        print(f"地图={n}*{m} 从{f}到{t}\n障碍:{b}")
        print(f"随机种子:{seed}")
        if not self.nograph:
            for i in range(n):
                for j in range(m):
                    self.X.append(i)
                    self.Y.append(j)
            self.C = [[0 for j in range(m)] for i in range(n)]
            se.set()
            plt.ion()
            self.gif = []

    def Search(self,
               threshold: float = -float("inf"),
               factor: float = -float("inf"),
               base: float = -float("inf")
             ):
        """
        搜索路径。
        """
        # 初始化:bfs表，设置起点，设置路线记录表
        self.trails = [node(self.f[0], self.f[1])]
        self.visited[self.f[0]][self.f[1]] = True
        self.path = [[None for j in range(self.m)] for i in range(self.n)]

        if threshold > -float("inf"):
            self.threshold = threshold
        if factor > -float("inf"):
            self.factor = factor
        if base > -float("inf"):
            self.base = base

        no = 0
        print(f"base:{self.base} factor:{self.factor} threshold:{self.threshold}")
        print("===训练开始===")

        while len(self.trails) > 0:
            # 当前level,初始值为当前最大的level
            now_level = max(self.trails, key=lambda x: x.level).level

            # 加入权重候选的节点序号表choices及其权重表weights
            choices = [i for i in range(len(self.trails)) if self.trails[i].level == now_level]
            weights = np.array([self.trails[i].weight for i in choices])

            # 如果序号表数量不够threshold则将更低一层的也加入进来，如果还是不够则继续扩大范围
            # 直至满足threshold的数量
            while (len(choices) < self.threshold) and (now_level > 0):
                now_level -= 1
                ex = [i for i in range(len(self.trails))
                      if self.trails[i].level == now_level]
                weights = np.concatenate((weights * self.base,
                                         np.array([self.trails[i].weight for i in ex])))
                choices.extend(ex)

            #随机抽取一个节点进行试探
            unitary = sum(weights)
            r = np.random.choice(choices, size=1, p=[i / unitary for i in weights])[0]
            no += 1
            print(
                f"{no}:搜索{(self.trails[r].x,self.trails[r].y)},偏好{self.trails[r].weight/unitary:.2f}({self.trails[r].weight:.2f}/{unitary:.2f}) , ", end="")
            x, y = self.trails[r].x, self.trails[r].y

            # ←
            if (x > 0) and (not self.visited[x - 1][y]):
                self.path[x - 1][y] = (x, y)
                print(f" ↓ 增加{(x-1,y)} ", end="")
                if self.__Ins(x - 1, y, self.trails[r]):
                    break

            # ↓
            if (y > 0) and (not self.visited[x][y - 1]):
                self.path[x][y - 1] = (x, y)
                print(f" ← 增加{(x,y-1)} ", end="")
                if self.__Ins(x, y - 1, self.trails[r]):
                    break

            # →
            if (x + 1 < self.n) and (not self.visited[x + 1][y]):
                self.path[x + 1][y] = (x, y)
                print(f" ↑ 增加{(x+1,y)} ", end="")
                if self.__Ins(x + 1, y, self.trails[r]):
                    break

            # ↑
            if (y + 1 < self.m) and (not self.visited[x][y + 1]):
                self.path[x][y + 1] = (x, y)
                print(f" → 增加{(x,y+1)} ", end="")
                if self.__Ins(x, y + 1, self.trails[r]):
                    break

            # 删去这个已试探节点
            del self.trails[r]
            print(f"队列长度:{len(self.trails)}")

            if not self.nograph:
                self.__frame()
                plt.gca().remove()
                plt.scatter(self.X, self.Y, s=6000 // self.n //
                            self.m, c=self.C, cmap='cubehelix')
                plt.pause(self.sleep)

        point = self.t
        route = []
        if len(self.trails) == 0:
            print("没有路径")
        else:
            print("已找到路径")
            # 从终点逆着寻找路径
            while point != None:
                route.append(point)
                point = self.path[point[0]][point[1]]
            print(f"输出路径:\n{'->'.join([str(i) for i in route[::-1]])}")
            print(f"总试探数:{no}\n队列最终长度:{len(self.trails)}\n最终权重表:")
            node_list = [f"{str((i.x,i.y))} {i.weight:.2f}*{self.base}^{i.level}" for i in sorted(self.trails, reverse=True)]
            if len(node_list) > 10:
                print("\n".join(node_list[:5]) +
                    f"\n...(省略{len(node_list)-10}个试探点)\n" \
                    + "\n".join(node_list[-5:]))
            else:
                print("\n".join(node_list))

        if not self.nograph:
            plt.plot([i[0] for i in route], [i[1] for i in route])
            plt.ioff()
            plt.show()



    def CLIEntrance(self,
                    n: int = 15,
                    m: int = 15,
                    f: tuple = (0, 0),
                    t: tuple = (14, 14),
                    b: list = [],
                    sleep: float = 0.1,
                    nograph: bool = False,
                    threshold: float = 5.0,
                    factor: float = 8.0,
                    base: float = 50.0):
        self.setMap(n, m, f, t, b, sleep, nograph)
        self.Search(threshold, factor, base)
        self.is_cli = True

if __name__ == "__main__":
    demo = MCTS()
    if len(sys.argv) > 1:
        Fire(demo.CLIEntrance)
        exit()
    demo.setMap(15, 15, (0,13), (14,1), [
        (1, 3), (2, 5), (3, 8), (5, 5), (4, 5),
        (9, 2), (6, 1), (6, 2), (6, 3), (6, 4),
        (7, 11), (9, 13), (12, 12), (6, 8), (9, 4),
        (13, 5), (12, 5), (11, 5), (11, 6), (11, 7),
        (2, 11), (4, 10), (1, 9), (1, 11), (1, 13),
        (13, 2), (14, 2), (10, 0), (12, 8), (13, 13),
        ], nograph=False)
    demo.Search()

