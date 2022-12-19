import pandas as pd
import os
from igraph import Graph
import numpy as np
from sklearn.linear_model import LinearRegression
import copy
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings("ignore")


def outcome(graph, lambda0, lambda1, lambda2, cluster_assignments, adjacency_matrix, degrees, ncps, stochastic):
    outcome = np.zeros(len(graph.vs)).astype(float)
    treatment_ncps = cluster_assignments * ncps
    control_ncps = ncps - treatment_ncps

    for i in range(0,3):
        outcome = lambda0 + lambda1*cluster_assignments + lambda2*np.sum(np.transpose(np.matmul(adjacency_matrix, np.diag(outcome)))/degrees, axis=0) + stochastic[:,i]

        intermediate = np.zeros(len(treatment_ncps))
        intermediate[np.where(treatment_ncps==1)] = lambda0
        intermediate[np.where(control_ncps==1)] = lambda0+lambda1
        intermediate = intermediate[np.where(ncps==1)]

        outcome[np.where(ncps==1)] = intermediate
    
    return outcome


def linear(adjacency_matrix, degrees, assignments, outcome):
    amt_treated = np.transpose(np.matmul(adjacency_matrix, assignments)) / degrees
    amt_treated[np.isnan(amt_treated)] = 0.0
    lm = LinearRegression()
    X = np.array([assignments, amt_treated])
    return lm.fit(np.transpose(X), outcome)


def ate(graph, index, lambda0, lambda1, lambda2, ncps, cluster_assignments, adjacency_matrix, degrees, transition_matrix, stochastic, pattern="random"):
    ate_without = lambda1+lambda2
    uncovered = 1 - np.matmul(ncps, adjacency_matrix) - ncps
    frac_uncovered = np.sum(uncovered)/(len(graph.vs))
    ncp_influence = np.sum(transition_matrix, axis=0)

    ncp_outcome = outcome(graph, lambda0, lambda1, lambda2, cluster_assignments, adjacency_matrix, degrees, ncps, stochastic)

    lin_gui = linear(adjacency_matrix, degrees, cluster_assignments, ncp_outcome)
    beta = lin_gui.coef_[0]
    gamma = lin_gui.coef_[1]
    ate_estimate_gui = beta+gamma
    
    ncp_influence = np.sum(ncp_influence[np.where(ncps==1)])/(len(graph.vs))

    return index, frac_uncovered, ncp_influence, ate_without, ate_estimate_gui, beta, gamma

def check_vertex_cover(adjacency_matrix, ncps): 
    adjacency_ncps = np.transpose( np.matmul(adjacency_matrix, np.transpose(ncps)) ) + np.transpose(ncps)
    return (adjacency_ncps > 0 ).sum() == np.shape(adjacency_matrix)[1]

def greedy(filename, graph, adjacency_matrix, gtype):
    dir_graph = Graph.Read_Edgelist(filename, directed=False)
    adjacency_matrix = np.array(Graph.get_adjacency(dir_graph).data) 
    degrees = Graph.outdegree(dir_graph)
    inverse = np.diag(np.reciprocal( np.array(degrees).astype(float) ))
    inverse[np.isinf(inverse)] = 0
    transition_matrix = np.transpose( np.matmul(inverse, np.array(adjacency_matrix.data)) )
    trans_sum = np.sum(transition_matrix, axis=0)

    adjacency = copy.deepcopy(adjacency_matrix)
    cover = np.zeros(len(adjacency))
    greedy = np.array([])
    while np.sum(cover) < len(adjacency):
        argmaxs = np.where(trans_sum==np.max(trans_sum)) 
        argmax = int(np.random.choice(argmaxs[0], 1))
        cover[adjacency[argmax]>0] = 1
        cover[argmax] = 1
        
        greedy = np.append(greedy, argmax)
        trans_matrix_temp = np.transpose( np.matmul(inverse, np.array(adjacency_matrix.data)) )
        for i in greedy.astype(int):
            trans_matrix_temp[i,:] = 0
            trans_matrix_temp[:,i] = 0
        trans_sum = np.sum(trans_matrix_temp, axis=0)

    return greedy 


def generate_ncps(filename, graph, adjacency_matrix, transition_matrix, pattern, maximum, gtype):
    ncps = np.zeros(len(graph.vs))
    if pattern == "random":
        if maximum:
            i = 1
            randlist = np.arange(0,len(graph.vs))
            np.random.shuffle(randlist)
            while not check_vertex_cover(adjacency_matrix, ncps):
                ncps[randlist[i]] = 1
                i += 1

    else: # worst-case (greedy)
        ncp_set = greedy(filename, graph, adjacency_matrix, gtype)
        ncp_set = np.unique(ncp_set)
        if maximum:
            for idx in ncp_set:
                ncps[int(idx)] = 1
    return ncps

def stochastic(n, rounds, sd):
    stoc = np.array([])
    for i in range(rounds):
        if i==0:
            stoc = np.transpose(np.random.normal(loc=0, scale=sd, size=n))
        else: 
            stoc = np.column_stack((stoc, np.transpose(np.random.normal(loc=0, scale=sd, size=n))))
    return stoc

def experiment(filename, pattern, gtype, lambda1, lambda2, lambda0=-1.5):
    graph = Graph.Read_Edgelist(filename, directed=False)
    # adjacency_matrix = np.loadtxt("/Users/kavery/workspace/correcting-ate-estimates/synthetic/clusters/fire500adj.csv", delimiter=",", dtype=int)
    # degrees = np.loadtxt("/Users/kavery/workspace/correcting-ate-estimates/synthetic/clusters/fire500degrees.csv", delimiter=",", dtype=int)
    degrees = Graph.degree(graph)
    vertices = len(graph.vs)
    adjacency_matrix = np.array(Graph.get_adjacency(graph).data)
    inverse = np.diag(np.reciprocal( np.array(degrees).astype(float) ))
    inverse[np.isinf(inverse)] = 0
    transition_matrix = np.transpose( np.matmul(inverse, np.array(adjacency_matrix.data)) )

    if gtype == "facebook":
        stoc = np.zeros((vertices, 3))
    else: 
        stoc = stochastic(vertices, 3, 0.1)

    # generate clusters
    clusters = graph.community_infomap()
    assignments = np.random.binomial(1, 0.5, len(clusters))

    cluster_assignments = np.zeros(len(graph.vs))
    for i in range(len(clusters)):
        for v in clusters[i]:
            cluster_assignments[v] = assignments[i]
    
    # ate
    non_ncp_ate = ate(graph, 0, lambda0, lambda1, lambda2, np.zeros(vertices), cluster_assignments, adjacency_matrix, degrees, transition_matrix, stoc, pattern)[4]

    if gtype == "facebook":
        ncps = np.zeros((vertices,))
        ncp_indices =  [0,107,348,414,686,698,1684,1912,3437,3980] # nodes used to make graph
        ncps[ncp_indices] = 1
    else:
        ncps = generate_ncps(filename, graph, adjacency_matrix, transition_matrix, gtype=gtype, pattern="greedy", maximum=True)
    total_ncps = np.sum(ncps==1)

    frac_ncps_ary = []
    normalized_diff_ary = []
    if pattern == "greedy":
        for i in range(total_ncps):
            greedy_ncps = np.zeros(len(graph.vs))
            greedy_ncps[np.where(ncps==1)[0][0:i]] = 1
            (index, frac_uncovered, ncp_influence, 
                ate_without, ate_estimate_gui, beta, 
                gamma) = ate(graph, i, lambda0, lambda1, lambda2, greedy_ncps, cluster_assignments, adjacency_matrix, degrees, transition_matrix, stoc, "greedy")
                
            frac_ncps = index/len(graph.vs)
            difference = non_ncp_ate - ate_estimate_gui
            normalized_diff = difference/non_ncp_ate
            if frac_ncps < 0.4:
                frac_ncps_ary.append(frac_ncps)
                normalized_diff_ary.append(normalized_diff)

    else: # random
        all_rand_ncps = np.zeros(len(graph.vs))
        rand_select = np.arange(0,len(graph.vs))
        np.random.shuffle(rand_select)
        rand_select = rand_select[:total_ncps-1]
        all_rand_ncps[rand_select] = 1
        for i in range(total_ncps):
            rand_ncps = np.zeros(len(graph.vs))
            rand_ncps[np.where(all_rand_ncps==1)[0][0:i]] = 1
            (index, frac_uncovered, ncp_influence, 
                ate_without, ate_estimate_gui, beta, 
                gamma) = ate(graph, i, lambda0, lambda1, lambda2, rand_ncps, cluster_assignments, adjacency_matrix, degrees, transition_matrix, stoc, "random")
            
            frac_ncps = index/len(graph.vs)
            normalized_diff = (non_ncp_ate - ate_estimate_gui)/non_ncp_ate
            if frac_ncps < 0.4:
                frac_ncps_ary.append(frac_ncps)
                normalized_diff_ary.append(normalized_diff)

    return frac_ncps_ary, normalized_diff_ary


if __name__=='__main__':
    cwd = os.getcwd()

    code = "small"
    for i in range(0,1):
        frac_ncps, normalized_diff = experiment(cwd+"/synthetic/"+code+"1000.net", gtype=code, lambda1=0.25, lambda2=1.0, pattern="greedy")
        plt.plot(frac_ncps, normalized_diff)
        plt.show()

        f = open("/Users/kavery/workspace/correcting-ate-estimates/results/test/"+str(i)+".csv", 'w')
        writer = csv.writer(f)
        writer.writerow(frac_ncps)
        writer.writerow(normalized_diff)
        f.close()
    print("done")
