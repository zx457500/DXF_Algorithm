# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import dxfgrabber
import math
import geatpy as ea # import geatpy
import numba
import multiprocessing as mp
from multiprocessing import Process,Manager
from multiprocessing import Pool as ProcessPool

class MyProblem(ea.Problem): # 继承Problem父类
    def __init__(self):
        self.cgl = np.array(self.dots(self.Closed_graphics_list(self.readDXF())))
        name = 'MyProblem' # 初始化name（函数名称，可以随意设置）
        M = 1 # 初始化M（目标维数）
        maxormins = [1] # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = len(self.cgl)  # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [1] * Dim # 决策变量下界
        ub = [Dim] * Dim # 决策变量上界
        lbin = [1] * Dim # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)    
        self.pool = ProcessPool(int(mp.cpu_count()))
        self.dic_dis = Manager().dict()
        # self.m = Manager().list()
        # self.n = Manager().list()
        # self.m.append(0)
        # self.n.append(0)

    def readDXF(self):# 读取DXF文件
        dxf = dxfgrabber.readfile("ClothesCAD120.dxf")
        # dxf = dxfgrabber.readfile("Drawing8.dxf")
        list_coordinate=[]       # 将直线坐标以（（起点坐标），（终点坐标））的形式放入列表中
        for e in dxf.entities:
            if e.dxftype == 'LINE':
                line = ['L',[(float('%.4f' % e.start[0]), float('%.4f' % e.start[1])), (float('%.4f' %e.end[0]),float('%.4f' %e.end[1]))]]
                list_coordinate.append(line)
            if e.dxftype == 'CIRCLE':
                C_x = float('%.4f' % e.center[0])       # 圆的圆心坐标（x,y,z）
                C_y = float('%.4f' % e.center[1])
                R = float('%.4f' % e.radius)            # 圆的半径
                """circle_coordinate = [标识位，圆心坐标，圆的半径，圆上四个点坐标]"""
                circle_coordinate = ('C' , (C_x , C_y) , R , [(C_x - R , C_y) , (C_x , C_y + R) , (C_x + R , C_y) , (C_x,C_y-R)])
                list_coordinate.append(circle_coordinate)
            if e.dxftype == 'ARC':
                arc_x = float('%.4f' % e.center[0])     # 圆弧圆心坐标x
                arc_y = float('%.4f' % e.center[1])     # 圆弧圆心坐标y
                arc_radius = float('%.4f' %  e.radius)  # 圆弧半径
                arc_start_angle = e.start_angle         # 圆弧起点角度
                arc_end_angle = e.end_angle             # 圆弧终点角度
                """圆弧起点坐标"""
                start_point = (float('%.4f' %(arc_x + arc_radius * math.cos(math.radians(arc_start_angle)))) , float('%.4f' % (arc_y + arc_radius * math.sin(math.radians(arc_start_angle)))))
                """圆弧终点坐标"""
                end_point = (float('%.4f' %(arc_x + arc_radius * math.cos(math.radians(arc_end_angle)))) , float('%.4f' %(arc_y + arc_radius * math.sin(math.radians(arc_end_angle)))))
                """arc_coordinate = [标识位，圆弧圆心坐标，圆弧半径，圆弧起点角度，圆弧终点角度，圆弧起点坐标，圆弧终点坐标]"""
                arc_coordinate = ['ARC' , (arc_x , arc_y) , arc_radius , arc_start_angle , arc_end_angle , (start_point , end_point)]
                list_coordinate.append(arc_coordinate)
        return list_coordinate
    
    def Closed_graphics_list(self,list_coordinate):  # 建立封闭图形
        a = []
        two_dimensional_list = []                               # 建立一个二维列表，保存形式：[[封闭图形1],[封闭图形2]....[封闭图形N]]                  
        """存放圆,并删除圆"""
        while len(list_coordinate) > 0:
            if list_coordinate[0][0] == 'C':
                two_dimensional_list.insert(0,list_coordinate[0])
                del list_coordinate[0]
            elif list_coordinate[0][0] == 'ARC' or list_coordinate[0][0] == 'L':
                a.append(list_coordinate[0])
                del list_coordinate[0]
                while len(list_coordinate) >= 0:
                    if abs(a[-1][-1][1][0] - a[0][-1][0][0]) > 0.001 or abs(a[-1][-1][1][1] - a[0][-1][0][1]) > 0.001:
                        for j in range(len(list_coordinate)):
                            if abs(list_coordinate[j][-1][0][0] - a[-1][-1][1][0]) < 0.001 and abs(list_coordinate[j][-1][0][1] - a[-1][-1][1][1]) < 0.001:  # 结点Nj的起点坐标 = 列表中最后一个结点的终点坐标
                                a.append(list_coordinate[j])
                                del list_coordinate[j]
                                break
                            elif abs(list_coordinate[j][-1][1][0] - a[-1][-1][1][0]) < 0.001 and abs(list_coordinate[j][-1][1][1] - a[-1][-1][1][1]) < 0.001:     # 结点Nj的终点坐标 = 列表中最后一个结点的终点坐标
                                tep_a = list_coordinate[j][-1][0]
                                tep_b = list_coordinate[j][-1][1]
                                tep_a , tep_b = tep_b , tep_a                   # 将结点起始坐标与终点坐标位置互换，保证封闭图形的画图顺序
                                list_coordinate[j][-1] = [tep_a,tep_b]
                                a.append(list_coordinate[j])
                                del list_coordinate[j]
                                break
                    elif abs(a[-1][-1][1][0] - a[0][-1][0][0]) < 0.001 and abs(a[-1][-1][1][1] - a[0][-1][0][1]) < 0.001:                           #最后一点与起点重合，则说明得到了封闭图形
                        two_dimensional_list.append(a)
                        a= []
                        break
        return two_dimensional_list

    def dots(self,two_dimensional_list):
        list = []
        list_temp = []
        for i in range(len(two_dimensional_list)):
            if two_dimensional_list[i][0] == 'C':
                list.append(two_dimensional_list[i][-1])
            elif two_dimensional_list[i][0][0] == 'ARC' or two_dimensional_list[i][0][0] == 'L':
                for j in range(len(two_dimensional_list[i])):
                    list_temp.append(two_dimensional_list[i][j][-1])
                while len(list_temp) > 0 :
                    list2 = []
                    for j in range(len(list_temp)):
                        list2.append(list_temp[j][0])
                    break
                list.append(list2)
                list_temp = []
        return list
    
    def Greedy_Algorithm(self,list_order): 
        list_order.append([(0.0,0.0)])
        list_order.insert(0,[(0.0,0.0)])
        a = []                              #距离矩阵 相邻的两个封闭图形各个点之间的距离（三维数组）
        N = len(list_order) - 1
        for x in range(N):
            dist_matrix = np.zeros((np.array(list_order[x]).shape[0],np.array(list_order[x+1]).shape[0]))  #建立距离矩阵（两个封闭图形之间各个点连线的距离）
            for y in range(len(list_order[x])):
                    for z in range(len(list_order[x+1])):
                        # qwe = list_order[x][y][0]
                        # qwe1 = list_order[x+1][z][0]
                        # qwe2 = list_order[x][y][1]
                        # qwe3 = list_order[x+1][z][1]
                        l = math.sqrt((list_order[x][y][0] - list_order[x+1][z][0])**2 + (list_order[x][y][1] - list_order[x+1][z][1])**2)
                        # dist_matrix[y,z] = self.dis(qwe,qwe1,qwe2,qwe3)
                        dist_matrix[y,z] = l 
            a.append(dist_matrix)
        """动态规划得到该顺序下的最短路径长度"""
        N = len(a) 
        Mark_temp = [] # 存放 第 i-1 个封闭图形中各个点 与 第 i 个图形 中各个点的距离的最小值 （一维数组）
        Mark = []     # 记忆功能，将Mark_temp存放进去，下一次循环时直接调用（即上一封闭图形各个点到终点的） （二维数组）               
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
 
    def aimFunc(self, pop):             # 目标函数
        x = pop.Phen    # 得到决策变量矩阵
        X = np.hstack([np.zeros((x.shape[0], 0)), x, np.zeros((x.shape[0], 0))]).astype(int)
        co_seq = self.cgl[X-1]
        result = self.pool.map_async(self.subAimFunc, co_seq)
        result.wait()
        pop.ObjV = np.array([result.get()]).T


    def subAimFunc(self,co_seq):               
        distance = self.Greedy_Algorithm(list(co_seq))
        return distance


    # def dis(self,qwe,qwe1,qwe2,qwe3):
    #     if (self.dic_dis.get(str(qwe)+ "," + str(qwe1)+ "," + str(qwe2)+ "," + str(qwe3))) == None:
    #         self.dic_dis[str(qwe)+ "," + str(qwe1)+ "," + str(qwe2)+ "," + str(qwe3)] = (qwe - qwe1)**2 + (qwe2 - qwe3)**2
    #         l = self.dic_dis[str(qwe)+ "," + str(qwe1)+ "," + str(qwe2)+ "," + str(qwe3)]
    #         self.m[0] += 1 
    #     else:
    #         l = self.dic_dis[str(qwe)+ "," + str(qwe1)+ "," + str(qwe2)+ "," + str(qwe3)]
    #         self.n[0] += 1 
    #     return l
        
    def __getstate__(self):                 #不加要出错
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state) 

    def Path(self,list_order):
        list_order.append([(0.0,0.0)])
        list_order.insert(0,[(0.0,0.0)]) 
        a = []                              #距离矩阵  所有封闭图形按顺序
        N = len(list_order) - 1
        for x in range(N):
            dist_matrix = np.zeros((np.array(list_order[x]).shape[0],np.array(list_order[x+1]).shape[0]))  #建立距离矩阵（两个封闭图形之间各个点连线的距离）
            for y in range(len(list_order[x])):
                    for z in range(len(list_order[x+1])):
                        l = math.sqrt((list_order[x][y][0] - list_order[x+1][z][0])**2 + (list_order[x][y][1] - list_order[x+1][z][1])**2)
                        dist_matrix[y,z] = l
            a.append(dist_matrix)
        N = len(a)
        Cost_temp = []
        Path_temp = []
        path = []
        distance_Cost = []
        distance_Mark = []
        Mark = []
        Cost_Closed_temp = []                                         
        for i in range(N-1,-1,-1):                                                # 自底而上进行计算
            if i == N-1:                                                          # N-1 = 6, 即 a 中最后一个距离矩阵
                Mark_temp = [a[i][j][0] for j in range(len(a[i]))]
                distance_Mark.append(Mark_temp)
            else:                                                                 # 当 1 < i < 6 时
                for j in range(len(a[i])):                                        # j 表示在该封闭图形的第 j 个点
                    Cost_temp = [a[i][j][k] + Mark_temp[k] for k in range(len(a[i][j]))]   # Cost_temp存放封闭图形一点到下一个封闭图形所有点的距离
                    distance_Cost.append(min(Cost_temp))
                    Cost_Closed_temp.append([min(Cost_temp) , Cost_temp.index(min(Cost_temp))]) #存放 这一点到下一个封闭图形中所有距离的最小值，和最小值所对应的下一个封闭图形上的点的索引位置
                    Cost_temp = []
                Path_temp.append(Cost_Closed_temp)
                Mark.append(distance_Cost)
                Mark_temp = []
                Mark_temp = [Cost_Closed_temp[w][0] for w in range(len(Cost_Closed_temp))]
                Cost_Closed_temp = []
                distance_Cost = []
        for i in range(len(Path_temp)-1,-1,-1):
            if i == len(Path_temp)-1:
                a = Path_temp[len(Path_temp)-1][0][1]
                path.append(a)
            else:
                path.append(Path_temp[i][a][1])
                a = Path_temp[i][a][1]
        Path = []
        path.append(0)
        path.insert(0,0)
        total_distance  = Mark[-1][0]
        Path = [list_order[i][path[i]] for i in range(len(path))]
        print('最短路程为：%s'%(total_distance))
        return Path

    def drawing_cgl(self,two_dimensional_list):
        start_x = []
        start_y = []
        for i in range(len(two_dimensional_list)):
            if two_dimensional_list[i][0] == 'C':
                theta = np.linspace(0, 2*np.pi,800)
                x,y = two_dimensional_list[i][1][0] + 0.5 * two_dimensional_list[i][2] * np.cos(theta)*2, two_dimensional_list[i][1][1] + 0.5 * two_dimensional_list[i][2] * np.sin(theta)*2
                plt.plot(x, y, color='black', linewidth=1.0)
            else:
                for z in range(len(two_dimensional_list[i])):
                    if two_dimensional_list[i][z][0] =='ARC':
                        if two_dimensional_list[i][z][3] > two_dimensional_list[i][z][4]:
                            theta = np.linspace(math.radians(two_dimensional_list[i][z][3])-2*np.pi , math.radians(two_dimensional_list[i][z][4]),800)
                        else:
                            theta = np.linspace(math.radians(two_dimensional_list[i][z][3]) , math.radians(two_dimensional_list[i][z][4]),800)
                        x,y = two_dimensional_list[i][z][1][0] + 0.5 * two_dimensional_list[i][z][2] * np.cos(theta)*2, two_dimensional_list[i][z][1][1] + 0.5 * two_dimensional_list[i][z][2] * np.sin(theta)*2
                        plt.plot(x, y, color='black', linewidth=1.0)
                    if two_dimensional_list[i][z][0] == 'L':
                        start_x.append(two_dimensional_list[i][z][1][0][0])
                        start_x.append(two_dimensional_list[i][z][1][1][0])
                        start_y.append(two_dimensional_list[i][z][1][0][1])
                        start_y.append(two_dimensional_list[i][z][1][1][1])
                        plt.plot(start_x, start_y,"b-", color='black', linewidth=1.0)
                        start_x = []
                        start_y = [] 


    def drawing(self,best_journey):
        list_order = []
        for j in range(len(best_journey)):
            list_order.append(self.cgl[int(best_journey[j])-1])
        Greedy_path = self.Path(list_order)
        x_coor = []
        y_coor = []
        for i in range(len(Greedy_path)):
            x_coor.append(Greedy_path[i][0])
            y_coor.append(Greedy_path[i][1])        
        plt.plot(x_coor, y_coor, c = 'black')
        plt.plot(x_coor, y_coor, 'o', c = 'black')