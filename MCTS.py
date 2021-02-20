# 用 双向BFS+Monte Carlo Tree Search解决 最短路径问题
# by 1mlightyears@gmail.com
# 20210215
import numpy as np
import matplotlib.pyplot as plt
from fire import Fire
from time import time
from copy import deepcopy
import seaborn as se

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
        p = - np.log(1 - d_ / d / 2) if d != 0 else max(self.weight)
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

    def MCTS(self):
        self.bfs = [tuple(self.f)]
        self.visited = deepcopy(self.grid)
        self.path = [[None for j in range(self.m)] for i in range(self.n)]
        self.weight = [1]
        self.s = 1
        no = 0
        #_ = np.pi - 3
        print("训练开始")
        while len(self.weight) > 0:
            r = np.random.choice(range(len(self.bfs)), size=1, p=[i / self.s for i in self.weight])[0]
            no += 1
            print(f"{no}:搜索{self.bfs[r]},偏好{self.weight[r]/self.s:.2f}({self.weight[r]:.2f}/{self.s:.2f}) , ", end="")
            x, y = self.bfs[r]

            #↑
            if (x - 1 > 0) and (not self.visited[x - 1][y]):
                self.path[x - 1][y] = (x, y)
                print(f" ↑ 增加{(x-1,y)} ", end="")
                if self.ins(x - 1, y, x, y):
                    break

            #←
            if (y - 1 > 0) and (not self.visited[x][y - 1]):
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
            self.frame()
            plt.gca().remove()
            plt.scatter(self.X, self.Y, s=6000 // self.n //
                        self.m, c=self.C, cmap='cubehelix')
            plt.pause(self.sleep)
        if len(self.weight) == 0:
            print("没有路径")
        else:
            print("已找到路径")
        point = self.t
        route=[]
        while point != None:
            route.append(point)
            point = self.path[point[0]][point[1]]
        print(f"输出路径:\n{'->'.join([str(i) for i in route[::-1]])}")
        print(f"总试探数:{no}\n队列最终长度:{len(self.bfs)}\n最终权值表:")
        print("\n".join([str((self.bfs[i], self.weight[i])) for i in range(len(self.bfs))]))
        plt.scatter([i[0] for i in route], [i[1] for i in route], s=6000 // self.n //
                    self.m, c=[0 for i in route], cmap='cubehelix')
        plt.ioff()
        plt.show()


#Fire(demo().data)
demo().data(10,10,(0, 0),(8, 9),[(1, 3), (2, 5), (3, 8), (5, 5), (4, 5), (9, 2), (6, 1), (6, 2), (6, 3), (6, 4)])
