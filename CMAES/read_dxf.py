import dxfgrabber
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import numba
class DXF_read():
    def __init__(self,dxf_doc):
        self.dxf_doc = dxf_doc
        self.two_dimensional_list = self.readDXF()
        self.list = self.dots()
    
    @numba.jit    
    def readDXF(self):                                      # 读取DXF文件，将DXF文件中的直线，圆，圆弧信息保存在list列表中
        list_coordinate=[]
        for e in self.dxf_doc.entities:
            if e.dxftype == 'LINE':
                # 直线保存形式: [ 标识位 ，[(起点坐标)，(终点坐标)]]
                line = ['L',[(float('%.4f' % e.start[0]), float('%.4f' % e.start[1])), (float('%.4f' %e.end[0]),float('%.4f' %e.end[1]))]]
                list_coordinate.append(line)
            if e.dxftype == 'CIRCLE':
                C_x = float('%.4f' % e.center[0])       # 圆的圆心坐标（x,y）
                C_y = float('%.4f' % e.center[1])
                R = float('%.4f' % e.radius)            # 圆的半径
                # 圆保存形式：circle_coordinate = [标识位，圆心坐标，圆的半径，圆上四个点坐标]"""
                circle = ('C' , (C_x , C_y) , R , [(C_x - R , C_y) , (C_x , C_y + R) , (C_x + R , C_y) , (C_x,C_y-R)])
                list_coordinate.append(circle)
            if e.dxftype == 'ARC':
                arc_x = float('%.4f' % e.center[0])     # 圆弧圆心坐标x
                arc_y = float('%.4f' % e.center[1])     # 圆弧圆心坐标y
                arc_radius = float('%.4f' %  e.radius)  # 圆弧半径
                arc_start_angle = e.start_angle         # 圆弧起点角度
                arc_end_angle = e.end_angle             # 圆弧终点角度
                start_point = (float('%.4f' %(arc_x + arc_radius * math.cos(math.radians(arc_start_angle)))) , float('%.4f' % (arc_y + arc_radius * math.sin(math.radians(arc_start_angle))))) 
                end_point = (float('%.4f' %(arc_x + arc_radius * math.cos(math.radians(arc_end_angle)))) , float('%.4f' %(arc_y + arc_radius * math.sin(math.radians(arc_end_angle)))))
                #圆弧保存形式：arc_coordinate = [标识位，圆弧圆心坐标，圆弧半径，圆弧起点角度，圆弧终点角度，圆弧起点坐标，圆弧终点坐标]"""
                arc_coordinate = ['ARC' , (arc_x , arc_y) , arc_radius , arc_start_angle , arc_end_angle , (start_point , end_point)]
                list_coordinate.append(arc_coordinate)
        a = []                                              # 临时列表，保存一个封闭图形中的所有信息
        two_dimensional_list = []                           # 建立一个二维列表，保存形式：[[封闭图形1],[封闭图形2]....[封闭图形N]]                  
        while len(list_coordinate) > 0:
            if list_coordinate[0][0] == 'C':                # 保存圆的信息，并删除list_coordinate中圆的信息
                two_dimensional_list.insert(0,list_coordinate[0])
                del list_coordinate[0]
            elif list_coordinate[0][0] == 'ARC' or list_coordinate[0][0] == 'L':
                a.append(list_coordinate[0])                # 保存其他封闭图形信息时，首先将list_coordinate中第一个图元信息拿出来
                del list_coordinate[0]                     
                while len(list_coordinate) >= 0:            # 在list_coordinate中找 与第一个图元信息 在同一个封闭图形中的 其他图元信息
                    if abs(a[-1][-1][1][0] - a[0][-1][0][0]) > 0.001 or abs(a[-1][-1][1][1] - a[0][-1][0][1]) > 0.001:    # 判断： 在a列表中的最后一个图元信息的终点坐标 是否 与第一个图元信a息的起点坐标相重合，精度为0.001
                        for j in range(len(list_coordinate)):
                            # 结点Nj的起点坐标 = 列表中最后一个结点的终点坐标
                            if abs(list_coordinate[j][-1][0][0] - a[-1][-1][1][0]) < 0.001 and abs(list_coordinate[j][-1][0][1] - a[-1][-1][1][1]) < 0.001:
                                a.append(list_coordinate[j])
                                del list_coordinate[j]
                                break
                            # 结点Nj的终点坐标 = 列表中最后一个结点的终点坐标
                            elif abs(list_coordinate[j][-1][1][0] - a[-1][-1][1][0]) < 0.001 and abs(list_coordinate[j][-1][1][1] - a[-1][-1][1][1]) < 0.001:
                                tep_a = list_coordinate[j][-1][0]
                                tep_b = list_coordinate[j][-1][1]
                                tep_a , tep_b = tep_b , tep_a          # 将结点起始坐标与终点坐标位置互换，保证封闭图形的画图顺序
                                list_coordinate[j][-1] = [tep_a,tep_b]
                                a.append(list_coordinate[j])
                                del list_coordinate[j]
                                break
                    elif abs(a[-1][-1][1][0] - a[0][-1][0][0]) < 0.001 and abs(a[-1][-1][1][1] - a[0][-1][0][1]) < 0.001: 
                        two_dimensional_list.append(a)       # 在a列表中的最后一个图元信息的终点坐标 与 第一个图元信息的起点坐标重合，则说明得到了封闭图形
                        a= []
                        break
        return two_dimensional_list
    
    @numba.jit
    def dots(self):  # 放置N个封闭图形的每个结点：[[1A,1B,1C....] , [2A,2B,2C,....], ......[NA,NB,NC.....]] 
        list = []
        list_temp = []
        for i in range(len(self.two_dimensional_list)):
            if self.two_dimensional_list[i][0] == 'C':
                list.append(self.two_dimensional_list[i][-1])
            elif self.two_dimensional_list[i][0][0] == 'ARC' or self.two_dimensional_list[i][0][0] == 'L':
                for j in range(len(self.two_dimensional_list[i])):
                    list_temp.append(self.two_dimensional_list[i][j][-1])
                while len(list_temp) > 0 :
                    list2 = []
                    for j in range(len(list_temp)):
                        list2.append(list_temp[j][0])
                    break
                list.append(list2)
                list_temp = []
        return list
    
    @numba.jit
    def Continuity(self):    # 将dots 变成 [2，6，4，1，3，5] 类型，作为cmaes基因进行排序
        L=[random.randint(0,10000) for _ in range(len(self.list))]
        Gene = np.argsort(L)+1
        return Gene
   

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
        Path = [list_order[i][path[i]] for i in range(len(path))]
        return Path
    
    
    @numba.jit
    def drawing_cgl(self):
        start_x = []
        start_y = []
        for i in range(len(self.two_dimensional_list)):
            if self.two_dimensional_list[i][0] == 'C':
                theta = np.linspace(0, 2*np.pi,800)
                x,y = self.two_dimensional_list[i][1][0] + 0.5 * self.two_dimensional_list[i][2] * np.cos(theta)*2, self.two_dimensional_list[i][1][1] + 0.5 * self.two_dimensional_list[i][2] * np.sin(theta)*2
                plt.plot(x, y, color='black', linewidth=1.0)
            else:
                for z in range(len(self.two_dimensional_list[i])):
                    if self.two_dimensional_list[i][z][0] =='ARC':
                        if self.two_dimensional_list[i][z][3] > self.two_dimensional_list[i][z][4]:
                            theta = np.linspace(math.radians(self.two_dimensional_list[i][z][3])-2*np.pi , math.radians(self.two_dimensional_list[i][z][4]),800)
                        else:
                            theta = np.linspace(math.radians(self.two_dimensional_list[i][z][3]) , math.radians(self.two_dimensional_list[i][z][4]),800)
                        x,y = self.two_dimensional_list[i][z][1][0] + 0.5 * self.two_dimensional_list[i][z][2] * np.cos(theta)*2, self.two_dimensional_list[i][z][1][1] + 0.5 * self.two_dimensional_list[i][z][2] * np.sin(theta)*2
                        plt.plot(x, y, color='black', linewidth=1.0)
                    if self.two_dimensional_list[i][z][0] == 'L':
                        start_x.append(self.two_dimensional_list[i][z][1][0][0])
                        start_x.append(self.two_dimensional_list[i][z][1][1][0])
                        start_y.append(self.two_dimensional_list[i][z][1][0][1])
                        start_y.append(self.two_dimensional_list[i][z][1][1][1])
                        plt.plot(start_x, start_y,"b-", color='black', linewidth=1.0)
                        start_x = []
                        start_y = [] 

    @numba.jit
    def drawing(self,best_routes):
        Route = best_routes[0]
        list_order = []
        for j in range(len(Route)):
            list_order.append(self.list[Route[j]])
        path = self.Path(list_order)
        x_coor = []
        y_coor = []
        for i in range(len(path)):
            x_coor.append(path[i][0])
            y_coor.append(path[i][1])        
        plt.plot(x_coor, y_coor, c = 'black')
        plt.plot(x_coor, y_coor, 'o', c = 'black')