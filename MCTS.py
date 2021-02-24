# 用 双向BFS+Monte Carlo Tree Search解决 最短路径问题
# by 1mlightyears@gmail.com
# 20210215

import numpy as np
import matplotlib.pyplot as plt
from fire import Fire
from time import time
from copy import deepcopy
import seaborn as se
from numba import jit

class demo:
    def data(self,
        n: int,
        m: int,
        f: tuple,
        t: tuple,
        b: list = [],
        sleep: float = 0.1,
        nograph: bool = False):

        self.grid = np.array([[False for j in range(m)] for i in range(n)])
        self.n = n
        self.m = m
        self.f = f
        self.t = t
        self.sleep = sleep
        self.nograph = nograph
        for i in b:
            self.grid[i[0]][i[1]] = True
        seed = int(time()) % (2 << 32 - 1)
        print(f"随机种子:{seed}")
        np.random.seed(seed)

        self.rec = {}
        self.min_threshold = 0.000001
        self.primitive = 2

        self.X, self.Y= [], []
        for i in range(n):
            for j in range(m):
                self.X.append(i)
                self.Y.append(j)
        self.C = [[0 for j in range(m)] for i in range(n)]
        print(f"地图={n}*{m} 从{f}到{t}\n障碍:{b}")
        if not self.nograph:
            se.set()
            plt.ion()
        self.MCTS()

    def ins(self, x, y, fx, fy):
        if (x, y) == self.t:
            return True
        d, d_ = (abs(self.t[0] - x) + abs(self.t[1] - y)), \
            (abs(self.t[0] - fx) + abs(self.t[1] - fy))

        if min(self.weight) < self.min_threshold:
            #如果太小 就扩大
            if d in self.rec:
                p = self.rec[d]
            elif d_ in self.rec:
                fac=np.log(d_)
                p = self.rec[d_] * fac if d < d_ else self.rec[d_] / fac
            else:
                p = self.primitive
        else:
            #不够小 就缩小
            p = self.rec[d_] if d_ in self.rec else self.primitive
            fac = np.log1p(d_/1.5)**(d_ - d)
            self.weight = list(map(lambda x: x / fac, self.weight))
            self.s = sum(self.weight)

        self.rec[d] = p

        self.bfs.append((x, y))
        self.visited[x][y] = True
        self.weight.append(p)
        self.s += p
        return False

    def frame(self):
        for i in range(self.n):
            for j in range(self.m):
                self.C[i][j] = 2 if self.grid[i][j] else 3 if (i, j) in self.bfs\
                    else 4 if self.visited[i][j] else 5
        self.C[self.f[0]][self.f[1]] = 0
        self.C[self.t[0]][self.t[1]] = 1

    #@jit(nopython=True)
    def MCTS(self):
        self.bfs = [tuple(self.f)]
        self.visited = deepcopy(self.grid)
        self.visited[self.f[0]][self.f[1]] = True
        self.path = [[None for j in range(self.m)] for i in range(self.n)]
        self.weight = [1]
        self.s = 1
        no = 0
        print("训练开始")
        while len(self.weight) > 0:
            r = np.random.choice(range(len(self.bfs)), size=1, p=[i / self.s for i in self.weight])[0]
            no += 1
            print(f"{no}:搜索{self.bfs[r]},偏好{self.weight[r]/self.s:.2f}({self.weight[r]:.2f}/{self.s:.2f}) , ", end="")
            x, y = self.bfs[r]

            #↑
            if (x > 0) and (not self.visited[x - 1][y]):
                self.path[x - 1][y] = (x, y)
                print(f" ↑ 增加{(x-1,y)} ", end="")
                if self.ins(x - 1, y, x, y):
                    break

            #←
            if (y > 0) and (not self.visited[x][y - 1]):
                self.path[x][y - 1] = (x, y)
                print(f" ← 增加{(x,y-1)} ", end="")
                if self.ins(x, y - 1, x, y):
                    break

            #↓
            if (x + 1 < self.n) and (not self.visited[x + 1][y]):
                self.path[x + 1][y] = (x, y)
                print(f" ↓ 增加{(x+1,y)} ", end="")
                if self.ins(x + 1, y, x, y):
                    break

            #→
            if (y + 1 < self.m) and (not self.visited[x][y + 1]):
                self.path[x][y + 1] = (x, y)
                print(f" → 增加{(x,y+1)} ", end="")
                if self.ins(x, y + 1, x, y):
                    break

            del self.bfs[r]
            self.s -= self.weight[r]
            del self.weight[r]
            print(f"队列长度:{len(self.bfs)}")
            if not self.nograph:
                self.frame()
                plt.gca().remove()
                plt.scatter(self.X, self.Y, s=6000 // self.n //
                            self.m, c=self.C, cmap='cubehelix')
                plt.pause(self.sleep)
        point = self.t
        route = []
        if len(self.weight) == 0:
            print("没有路径")
        else:
            print("已找到路径")
            while point != None:
                route.append(point)
                point = self.path[point[0]][point[1]]
            print(f"输出路径:\n{'->'.join([str(i) for i in route[::-1]])}")
            print(f"总试探数:{no}\n队列最终长度:{len(self.bfs)}\n最终权值表:")
            self.weight = [str(round(i * 100000) / 100000)
                        for i in sorted(self.weight, reverse=True)]
            self.bfs = [str(i) for i in sorted(self.bfs, reverse=True)]
            if len(self.weight) > 10:
                print("\n".join([self.bfs[i]+" "+self.weight[i] for i in range(5)]) +
                    f"\n...(省略{len(self.bfs)-10}个试探点)\n" \
                    + "\n".join([self.bfs[-i - 1] + " " + self.weight[-i - 1] for i in range(5)][::-1]))
            else:
                print("\n".join([self.bfs[i]+" "+self.weight[i]
                                for i in range(len(self.weight))]))
        if not self.nograph:
            plt.scatter([i[0] for i in route], [i[1] for i in route], s=6000 // self.n //
                        self.m, c=[0 for i in route], cmap='cubehelix')
            plt.ioff()
            plt.show()


#Fire(demo().data)
demo().data(15, 15, (0,0), (14,14), [
    (1, 3), (2, 5), (3, 8), (5, 5), (4, 5),
    (9, 2), (6, 1), (6, 2), (6, 3), (6, 4),
    (7, 11), (9, 13), (12, 12), (6, 8), (9, 4),
    (13, 5), (12, 5), (11, 5), (11, 6), (11, 7),
    (2, 11), (4, 10), (1, 9), (1, 11), (1, 13),
    (13, 2), (14, 1), (10, 0), (12, 8), (13, 13),
    ],nograph=False)

