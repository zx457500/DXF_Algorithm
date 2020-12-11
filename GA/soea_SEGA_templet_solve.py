# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import dxfgrabber
import math
import geatpy as ea 
from MyProblem import MyProblem   

if __name__ == '__main__':  
    """================================实例化问题对象=======x====================="""
    p = MyProblem() # 生成问题对象
    d = p.Closed_graphics_list(p.readDXF())
    cgl=p.dots(d)
    """==================================种群设置==============================="""
    Encoding = 'P'        # 编码方式
    NIND = 90             # 种群规模
    Field = ea.crtfld(Encoding, p.varTypes, p.ranges, p.borders) # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    myAlgorithm = ea.soea_SEGA_templet(p, population) # 实例化一个算法模板对象
    # myAlgorithm = ea.soea_GGAP_SGA_templet(p, population) # 实例化一个算法模板对象
    # myAlgorithm = ea.soea_steadyGA_templet(p, population) # 实例化一个算法模板对象
    # myAlgorithm = ea.soea_studGA_templet(p, population) # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 1600       # 最大进化代数
    myAlgorithm.mutOper.Pm = 0.5 # 变异概率
    myAlgorithm.recOper = ea.Xovox(XOVR = 0.6)  # 交叉算子
    myAlgorithm.recOper.Parallel=True 
    myAlgorithm.drawing = 1 
    """===========================调用算法模板进行种群进化======================="""
    [population, obj_trace, var_trace] = myAlgorithm.run() # 执行算法模板
    population.save() # 把最后一代种群的信息保存到文件中
    p.pool.close()
    """===============================输出结果及绘图============================"""
    # 输出结果
    best_gen = np.argmin(p.maxormins * obj_trace[:, 1]) # 记录最优种群个体是在哪一代
    best_ObjV = np.min(obj_trace[:, 1])
    print('最佳路线为：')
    best_journey = np.hstack([var_trace[best_gen, :]])
    for i in range(len(best_journey)):
        print(int(best_journey[i]), end = ' ')
    print('有效进化代数：%s'%(obj_trace.shape[0]))
    print('最优的一代是第 %s 代'%(best_gen + 1))
    print('时间已过 %s 秒'%(myAlgorithm.passTime))
    p.drawing(best_journey) 
    p.drawing_cgl(d)
    plt.show()