import numpy as np
import matplotlib.pyplot as plt
import json
from itertools import cycle, islice, dropwhile


#################################
#### General-purpose tools ######
#################################

def read_data(filepath, prob_name, prob_idx):
    with open(filepath, "r") as data:
        d = json.load(data)
    return d[prob_name], d['f_bias'][prob_idx]


# def print_results(time_elapsed, prob, pop=None, archi=None):
#     if pop:
#         nevals = pop.problem.get_fevals()
#         champion_f = pop.champion_f[0]
#         champion_x = pop.champion_x
#     elif archi:
#         champions_f = archi.get_champions_f()
#         champions_x = archi.get_champions_x()
#         champion_idx = np.argmin(champions_f)
#         champion_f = champions_f[champion_idx][0]
#         champion_x = champions_x[champion_idx]
#         nevals = 0
#         for isl in archi:
#             pop = isl.get_population()
#             nevals += pop.problem.get_fevals()

#     print()
#     print("Computation time: {:.0f}m {:.1f}s".format(time_elapsed // 60, time_elapsed % 60))
#     print("Number of function evaluations: {:d}\n".format(nevals))

#     print("Best fitness:\t{:.3f}".format(champion_f))
#     print("Best solution:", end='\t\t')
#     for x in champion_x[:4]:
#         print("{:.2f}".format(x), end='  ')
#     print("...", end='  ')
#     for x in champion_x[-3:]:
#         print("{:.2f}".format(x), end='  ')
#     print("\n")

#     udp = prob.extract(object)
#     optimum = udp.o
#     bias = udp.bias
#     dist = np.sqrt(np.sum((champion_x - optimum)**2))
#     delta = abs(champion_f - bias)

#     print("Global minimum:\t{:.3f}".format(bias))
#     print("Global optimal point:", end='\t')
#     for x in optimum[:4]:
#         print("{:.2f}".format(x), end='  ')
#     print("...", end='  ')
#     for x in optimum[-3:]:
#         print("{:.2f}".format(x), end='  ')
#     print("\n")

#     print("Difference with optimum:\t{:.2e}".format(delta))
#     print("Distance to optimal point:\t{:.2e}".format(dist))
#     print("\n")


def plot_fitness(logs, minimum, prob_name, label, logscale=True, save=False, savepath=None):
    fig = plt.figure(figsize=(7,5))
    # plt.title("convergence graph for {}".format(prob_name))
    for log, lbl in zip(logs, label):
        # plt.plot(log[:,1], log[:,2] - minimum, label=lbl)
        plt.plot(log[:,1], log[:,2], label=lbl)
    plt.xlabel("function evaluations")
    plt.ylabel("f(x)")
    plt.legend(loc='upper right')
    if logscale:
        plt.yscale('log')
    if save:
        fig.savefig(savepath)
    plt.show()


#################################
########### TSP tools ###########
#################################

def read_tsplib(filepath):
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if line[:9] == 'DIMENSION':
                dim = int(line.split(': ')[-1])
            elif line.rstrip() == 'NODE_COORD_SECTION':
                cities = np.loadtxt(filepath, delimiter=' ',
                        usecols=(1,2), skiprows=i+1, max_rows=dim)
                break
    return cities


def get_best_route(pop=None, archi=None):
    if pop:
        nevals = pop.problem.get_fevals()
        champion_f = pop.champion_f[0]
        champion_x = pop.champion_x
    elif archi:
        champions_f = archi.get_champions_f()
        champions_x = archi.get_champions_x()
        champion_idx = np.argmin(champions_f)
        champion_f = champions_f[champion_idx][0]
        champion_x = champions_x[champion_idx]
        nevals = 0
        for isl in archi:
            pop = isl.get_population()
            nevals += pop.problem.get_fevals()

    best_route = np.argsort(champion_x)

    return best_route, champion_f, nevals


def print_tsp_results(time_elapsed, nevals, best_route, best_fits):
    print("Computation time: {:.0f}m {:.1f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Number of function evaluations: {:d}\n".format(nevals))
    print("Length of best route found by EA:\t\t{:.0f}".format(best_fits[0]))
    print("Best route:", best_route)


def plot_route(cities, best_routes, title, figsize=(7,7), with_route_ea=True, save=False, savepath=None):
    fig = plt.figure(figsize=figsize)
    plt.scatter(x=cities[:,1], y=cities[:,0], s=10, c='C0', zorder=3)
    linestyles = ['-']
    colors = ['darkgray', 'red']
    labels = ['EA', 'EA + 3-opt']
    for i, (route, ls, c, lbl) in enumerate(zip(best_routes, linestyles, colors, labels)):
        if (not with_route_ea) and (i == 0):
            continue
        r = np.append(route, route[0])
        plt.plot(cities[r,1], cities[r,0], c=c, ls=ls, lw=1, label=lbl, zorder=i+1)
    plt.title(title)
    plt.legend(loc='best')
    plt.axis('equal')
    plt.tick_params(axis='both', which='both',
                    bottom=False, left=False,
                    labelbottom=False, labelleft=False)
    if save:
        fig.savefig(savepath)
    plt.show()


def two_opt(route, weights):
    dim = route.size
    best_route = route
    best_tour_length = sum(weights[i,j]
                        for i,j in zip(best_route, np.roll(best_route, 1)))
    print()
    print(" 2-opt exchanges \t fitness")
    improvement = True
    while improvement:
        improvement = False
        for i in range(dim-2):
            for j in range(i+2, dim-(i==0)):
                A, B = best_route[i], best_route[i+1]
                C, D = best_route[j], best_route[(j+1)%dim]
                delta = weights[A,C] + weights[B,D] \
                    - weights[A,B] - weights[C,D]
                if delta < 0:
                    best_route[i+1:j+1] = np.flip(best_route[i+1:j+1])
                    best_tour_length += delta
                    improvement = True
                    print("   {:3d} - {:3d} \t\t  {:.0f}".format(A, C, best_tour_length))
                    break
            if improvement:
                break
    return best_route, best_tour_length


def three_opt(route, weights):
    cases = ['case_1', 'case_2', 'case_3',
             'case_4', 'case_5', 'case_6', 'case_7']
    moves_cost = {'case_1': 0, 'case_2': 0,
                  'case_3': 0, 'case_4': 0, 'case_5': 0,
                  'case_6': 0, 'case_7': 0, 'case_8': 0}
    print()
    print("   N   \t 3-opt recombinations \t fitness")
    print(" ----- \t -------------------- \t -------")
    improved = True
    counter = 0
    best_route = route
    best_tour_length = sum(weights[i,j]
                        for i,j in zip(best_route, np.roll(best_route, 1)))
    while improved:
        improved = False
        for (i, j, k) in possible_segments(len(route)):
            # check all possible moves and save result into the dict
            for case in cases:
                moves_cost[case] = get_solution_cost_change(best_route, weights, case, i, j, k)
            # need minimum value of substraction of old route - new route
            best_return = max(moves_cost, key=moves_cost.get)
            if moves_cost[best_return] > 0:
                best_route = reverse_segments(best_route, best_return, i, j, k)
                best_tour_length -= moves_cost[best_return]
                counter += 1
                improved = True
                A, B, C = best_route[i], best_route[j], best_route[k % len(route)]
                print(" {:>4d}  \t   {:>3d} - {:>3d} - {:>3d}    \t  {:>5.0f}".format(counter, A, B, C, best_tour_length))
                break
    # to start with the same node -> need to cycle results
    cycled = cycle(best_route)
    skipped = dropwhile(lambda x: x != 0, cycled)
    sliced = islice(skipped, None, len(best_route))
    best_route = list(sliced)
    best_tour_length = sum(weights[i,j]
                        for i,j in zip(best_route, np.roll(best_route, 1)))
    return np.array(best_route), best_tour_length


def possible_segments(d):
    segments = ((i, j, k) for i in range(d)
                for j in range(i + 2, d-1)
                for k in range(j + 2, d - 1 + (i > 0)))
    return segments


def get_solution_cost_change(route, weights, case, i, j, k):
    """ Compare current solution with 7 possible 3-opt moves"""
    A, B, C = route[i - 1], route[i], route[j - 1]
    D, E, F = route[j], route[k - 1], route[k % len(route)]
    if case == 'case_1':
        # current solution ABC
        return 0
    elif case == 'case_2':
        return weights[A, B] + weights[E, F] \
            - (weights[B, F] + weights[A, E])
    elif case == 'case_3':
        return weights[C, D] + weights[E, F] \
            - (weights[D, F] + weights[C, E])
    elif case == 'case_4':
        return weights[A, B] + weights[C, D] + weights[E, F] \
           - (weights[A, D] + weights[B, F] + weights[E, C])
    elif case == 'case_5':
        return weights[A, B] + weights[C, D] + weights[E, F] \
            - (weights[C, F] + weights[B, D] + weights[E, A])
    elif case == 'case_6':
        return weights[B, A] + weights[D, C] \
            - (weights[C, A] + weights[B, D])
    elif case == 'case_7':
        return weights[A, B] + weights[C, D] + weights[E, F] \
            - (weights[B, E] + weights[D, F] + weights[C, A])
    elif case == 'case_8':
        return weights[A, B] + weights[C, D] + weights[E, F] \
            - (weights[A, D] + weights[C, F] + weights[B, E])


def reverse_segments(route, case, i, j, k):
    if (i - 1) < (k % len(route)):
        first_segment = np.concatenate((route[k% len(route):], route[:i]))
    else:
        first_segment = route[k % len(route):i]
    second_segment = route[i:j]
    third_segment = route[j:k]

    if case == 'case_1':
        # current solution ABC
        pass
    elif case == 'case_2':
        solution = np.concatenate((list(reversed(first_segment)),
                                   second_segment,third_segment))
    elif case == 'case_3':
        solution = np.concatenate((first_segment, second_segment,
                                   list(reversed(third_segment))))
    elif case == 'case_4':
        solution = np.concatenate((list(reversed(first_segment)),
                                   second_segment, list(reversed(third_segment))))
    elif case == 'case_5':
        solution = np.concatenate((list(reversed(first_segment)),
                                   list(reversed(second_segment)), third_segment))
    elif case == 'case_6':
        solution = np.concatenate((first_segment,
                                   list(reversed(second_segment)), third_segment))
    elif case == 'case_7':
        solution = np.concatenate((first_segment, list(reversed(second_segment)),
                                   list(reversed(third_segment))))
    elif case == 'case_8':
        solution = np.concatenate((list(reversed(first_segment)), list(reversed(second_segment)),
                                   list(reversed(third_segment))))
    return solution

