import read_dxf
import dxfgrabber
import math
import numpy as np
import numba 
from multiprocessing import Process,Manager
from functools import lru_cache
class dxf_func():
    def __init__(self,dots):
        self.list_order = dots

    def DP_Algorithm(self):  # 动态规划算法 计算该顺序下最短路径长度
        self.list_order.append([(0.0,0.0)])
        self.list_order.insert(0,[(0.0,0.0)]) 
        a = []                              #距离矩阵  所有封闭图形中各个结点之间的距离值
        N = len(self.list_order) - 1
        for x in range(N):
            dist_matrix = np.zeros((np.array(self.list_order[x]).shape[0],np.array(self.list_order[x+1]).shape[0]))  #建立距离矩阵（两个封闭图形之间各个点连线的距离）
            for y in range(len(self.list_order[x])):
                    for z in range(len(self.list_order[x+1])):
                        # qwe = self.list_order[x][y][0]
                        # qwe1 = self.list_order[x+1][z][0]
                        # qwe2 = self.list_order[x][y][1]
                        # qwe3 = self.list_order[x+1][z][1]
                        # dist_matrix[y,z] = self.dis(qwe,qwe1,qwe2,qwe3)
                        l = math.sqrt((self.list_order[x][y][0] - self.list_order[x+1][z][0])**2 + (self.list_order[x][y][1] - self.list_order[x+1][z][1])**2)
                        dist_matrix[y,z] = l
            a.append(dist_matrix)
        N = len(a) 
        Mark_temp = []                      # 存放 第 i-1 个封闭图形中各个点 与 第 i 个图形 中各个点的距离的最小值 （一维数组）
        Mark = []                           # 记忆功能，将Mark_temp存放进去，下一次循环时直接调用（即上一封闭图形各个点到终点的） （二维数组）
        for i in range(N-1,-1,-1):                                                # 自底而上进行计算
            if i == N-1:                                                          # N-1 即 为 a 中最后一个距离矩阵
                list_temp = [a[i][j][0] for j in range(len(a[i]))]              
                Mark.append(list_temp)                                              
            else:
                for j in range(len(a[i])):                                        # j 表示在该封闭图形的第 j 个点
                    Mark_temp.append(min([a[i][j][k] + Mark[-1][k] for k in range(len(a[i][j]))])) 
                Mark.append(Mark_temp)
                Mark_temp = []                                   # 清除上一次循环中各点距离最小值
        total_distance  = Mark[-1][0]                            # 最后一个数组的值就是起点到终点的最优路径长度
        return total_distance 

    # @lru_cache(maxsize=None)
    # def dis(self,qwe,qwe1,qwe2,qwe3):
    #     l = math.sqrt((qwe - qwe1)**2 + (qwe2 - qwe3)**2)
    #     return l