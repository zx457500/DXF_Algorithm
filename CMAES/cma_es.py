import read_dxf
import dxfgrabber
import math
import numpy as np
from dxf_func import dxf_func
import time
import pygmo as pg
import random
import tools
import matplotlib.pyplot as plt
from multiprocessing import Process,Manager

class gtsp():
    def __init__(self,dxf_doc):
        dxf = read_dxf.DXF_read(dxf_doc)  
        self.dots = dxf.dots()         # dots = [[a1,a2,a3,...,a],[b1,b2,b3,....,bn],...[m1,m2,m3,...,mn]],对dots进行基因排序
        self.Gene = dxf.Continuity()
        self.dim = np.array(self.dots).shape[0]
        self.dic_dis = Manager().dict()

    def fitness(self,Gene):
        u = np.argsort(Gene)
        list_order= []
        for i in range(Gene.shape[0]):
            list_order.append(self.dots[u[i]])
        dis = dxf_func(list_order)
        res = dis.DP_Algorithm()
        return [res]

    def get_bounds(self):
        return ([0] * self.dim, [1] * self.dim)
   
def main():
    dxf_doc = dxfgrabber.readfile("ClothesCAD120.dxf")
    # dxf_doc = dxfgrabber.readfile("Drawing8.dxf")
    dxf = read_dxf.DXF_read(dxf_doc) 
    SEED = 60
    prob = pg.problem(gtsp(dxf_doc))
    # algo = pg.algorithm(pg.cmaes(gen=6000, cc = -1, cs = -1, c1 = -1, cmu = -1, sigma0 = 2.3, ftol = 1e-11, xtol = 1e-6, memory = True, force_bounds = False,  seed=SEED))
    # algo = pg.algorithm(pg.de(gen =40000,F = 0.9,CR = 0.6,seed = SEED))
    algo = pg.algorithm(pg.pso(gen =50, omega = 0.8, eta1 = 2, eta2 = 2, max_vel = 0.5, variant = 5, neighb_type = 2, neighb_param = 8, memory = False, seed = SEED))
    
    algo.set_verbosity(1)

    archi = pg.archipelago(n=1, t=pg.fully_connected(), udi=pg.mp_island(),algo=algo, prob=prob, pop_size=100,r_pol=pg.fair_replace(), s_pol=pg.select_best(), seed=SEED)
        
    since = time.time()
    archi.evolve() 
    archi.wait()

    best_route_ea, best_fit_ea, nevals_ea = tools.get_best_route(archi=archi)

    time_elapsed = time.time() - since
    logs = []
    labels = []
    for i, isl in enumerate(archi):
        algo = isl.get_algorithm()
        
        # logs.append(np.array(algo.extract(pg.cmaes).get_log()))
        # logs.append(np.array(algo.extract(pg.de).get_log()))
        logs.append(np.array(algo.extract(pg.pso).get_log()))
        
        # labels.append('CMA-ES {:d}'.format(i+1))
        # labels.append('DE {:d}'.format(i+1))
        labels.append('PSO {:d}'.format(i+1))

    best_routes = [best_route_ea]
    best_fits = [best_fit_ea]
    tools.print_tsp_results(time_elapsed, nevals_ea, best_routes[0], best_fits)
    tools.plot_fitness(logs, 9352, " ", labels)
    
    dxf.drawing_cgl()
    dxf.drawing(best_routes)
    plt.show()


if __name__ == "__main__":
    main()
