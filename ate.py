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


def spillover_probs(ncps, assignments, adjacency_matrix, degrees, transition_matrix):
    non_ncp_treatment = assignments * (1-ncps)
    non_ncp_treatment_trans = np.transpose(non_ncp_treatment) / degrees
    non_ncp_treatment_trans[np.isnan(non_ncp_treatment_trans)] = 0
    non_ncp_treatment_neigh = np.transpose(np.matmul(adjacency_matrix, non_ncp_treatment_trans))
    
    non_ncp_control = (1-assignments) * (1-ncps)
    non_ncp_control_trans = np.transpose(non_ncp_treatment) / degrees
    non_ncp_control_trans[np.isnan(non_ncp_control_trans)] = 0
    non_ncp_control_neigh = np.transpose(np.matmul(adjacency_matrix, non_ncp_control_trans))
    
    ncp_treatment = assignments * ncps
    ncp_treatment_trans = np.transpose(ncp_treatment) / degrees
    ncp_treatment_trans[np.isnan(ncp_treatment_trans)] = 0
    ncp_treatment_neigh = np.transpose(np.matmul(adjacency_matrix, ncp_treatment_trans))
    
    ncp_control = ncps - ncp_treatment
    ncp_control_trans = np.transpose(ncp_treatment) / degrees
    ncp_control_trans[np.isnan(ncp_control_trans)] = 0
    ncp_control_neigh = np.transpose(np.matmul(adjacency_matrix, ncp_control_trans))
    
    adversarial_influence = np.sum(transition_matrix, axis=0)
    return non_ncp_treatment_neigh, non_ncp_control_neigh, ncp_treatment_neigh, ncp_control_neigh, adversarial_influence


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


def linear_ncp(non_ncp_treatment_neigh, ncp_treatment_neigh, ncp_control_neigh, assignments, outcome):
    lm = LinearRegression()
    X = np.array([assignments, non_ncp_treatment_neigh, ncp_treatment_neigh, ncp_control_neigh])
    return lm.fit(np.transpose(X), outcome)


def ate(graph, index, lambda0, lambda1, lambda2, ncps, cluster_assignments, adjacency_matrix, degrees, transition_matrix, stochastic, pattern="random"):
    ate_without = lambda1+lambda2
    uncovered = 1 - np.matmul(ncps, adjacency_matrix) - ncps
    frac_uncovered = np.sum(uncovered)/(len(graph.vs))
    (non_ncp_treatment_neigh, 
        non_ncp_control_neigh, 
        ncp_treatment_neigh, 
        ncp_control_neigh, 
        adversarial_influence) = spillover_probs(ncps, cluster_assignments, adjacency_matrix, degrees, transition_matrix)

    ncp_outcome = outcome(graph, lambda0, lambda1, lambda2, cluster_assignments, adjacency_matrix, degrees, ncps, stochastic)

    lin_ncp = linear_ncp(non_ncp_treatment_neigh, ncp_treatment_neigh, ncp_control_neigh, cluster_assignments, ncp_outcome)
    ate_estimate = lin_ncp.coef_[0]+lin_ncp.coef_[1]
    ate_ncp_bias = lin_ncp.coef_[2]-lin_ncp.coef_[3]

    lin_gui = linear(adjacency_matrix, degrees, cluster_assignments, ncp_outcome)
    beta = lin_gui.coef_[0]
    gamma = lin_gui.coef_[1]
    ate_estimate_gui = beta+gamma

    if pattern == "dominating" or index == 0:
        max_ncps = False
    else:
        max_ncps = True
    
    ncp_influence = np.sum(adversarial_influence[np.where(ncps==1)])/(len(graph.vs))

    return index, max_ncps, frac_uncovered, ncp_influence, ate_without, ate_ncp_bias, ate_estimate_gui, beta, gamma

def check_set(adjacency_matrix, ncps): # check this? 
    adjacency_ncps = np.transpose( np.matmul(adjacency_matrix, np.transpose(ncps)) ) + np.transpose(ncps)
    return (adjacency_ncps > 0 ).sum() == np.shape(adjacency_matrix)[1]

def greedy(filename, graph, adjacency_matrix, gtype, weight="influence"):
    dir_graph = Graph.Read_Edgelist(filename, directed=False)
    adjacency_matrix = np.array(Graph.get_adjacency(dir_graph).data) 
    if weight == "influence":
        degrees = Graph.outdegree(dir_graph)
        inverse = np.diag(np.reciprocal( np.array(degrees).astype(float) ))
        inverse[np.isinf(inverse)] = 0
        transition_matrix = np.transpose( np.matmul(inverse, np.array(adjacency_matrix.data)) )
        trans_sum = np.sum(transition_matrix, axis=0)
    else:
        dir_graph = Graph.Read_Edgelist(filename, directed=False)
        trans_sum = Graph.outdegree(dir_graph)

    adjacency = copy.deepcopy(adjacency_matrix)
    cover = np.zeros(len(adjacency))
    greedy = np.array([])
    while np.sum(cover) < len(adjacency):
        argmaxs = np.where(trans_sum==np.max(trans_sum)) 
        argmax = int(np.random.choice(argmaxs[0], 1))
        cover[adjacency[argmax]>0] = 1
        cover[argmax] = 1
        if gtype == "facebook" and len(greedy) >= 10:
            return greedy
        
        greedy = np.append(greedy, argmax)
        if weight == "influence":
            trans_matrix_temp = np.transpose( np.matmul(inverse, np.array(adjacency_matrix.data)) )
            for i in greedy.astype(int):
                trans_matrix_temp[i,:] = 0
                trans_matrix_temp[:,i] = 0
            trans_sum = np.sum(trans_matrix_temp, axis=0)
        else: 
            directed = np.array(Graph.get_adjacency(dir_graph).data) 
            directed[cover>0]=0 
            dir_graph = Graph.Adjacency(directed)
            trans_sum = Graph.outdegree(dir_graph)-cover
    return greedy 


def generate_ncps(filename, graph, adjacency_matrix, transition_matrix, pattern, weight, maximum, gtype):
    ncps = np.zeros(len(graph.vs))
    if pattern == "random":
        if weight == "influence":
            if maximum:
                while not check_set(adjacency_matrix, ncps):
                    print(transition_matrix[np.argsort(-1*transition_matrix)])
                    ncps[transition_matrix[np.argsort(-1*transition_matrix)]] = 1
        else:
            if maximum:
                i = 1
                randlist = np.arange(0,len(graph.vs))
                np.random.shuffle(randlist)
                while not check_set(adjacency_matrix, ncps):
                    ncps[randlist[i]] = 1
                    i += 1

    else: # worst-case (greedy)
        ncp_set = greedy(filename, graph, adjacency_matrix, gtype, weight=weight)
        ncp_set = np.unique(ncp_set)
        if maximum:
            for idx in ncp_set:
                ncps[int(idx)] = 1
    return ncps

def stochastic(n, steps, sd):
    stoc = np.array([])
    for i in range(steps):
        if i==0:
            stoc = np.transpose(np.random.normal(loc=0, scale=sd, size=n))
        else: 
            stoc = np.column_stack((stoc, np.transpose(np.random.normal(loc=0, scale=sd, size=n))))
    return stoc

def experiment(filename, pattern, weight, gtype, lambda1, lambda2, lambda0=-1.5):
    graph = Graph.Read_Edgelist(filename, directed=False)
    degrees = Graph.degree(graph)
    vertices = len(graph.vs)
    adjacency_matrix = np.array(Graph.get_adjacency(graph).data)
    inverse = np.diag(np.reciprocal( np.array(degrees).astype(float) ))
    inverse[np.isinf(inverse)] = 0
    transition_matrix = np.transpose( np.matmul(inverse, np.array(adjacency_matrix.data)) )

    # generate clusters
    clusters = graph.community_infomap()
    assignments = np.random.binomial(1, 0.5, len(clusters))

    cluster_assignments = np.zeros(len(graph.vs))
    for i in range(len(clusters)):
        for v in clusters[i]:
            cluster_assignments[v] = assignments[i]

    if gtype == "facebook":
        stoc = np.zeros((len(graph.vs), 3))
    else: 
        stoc = stochastic(vertices, 3, 0.1)
    
    # ate
    non_ncp_ate = ate(graph, 0, lambda0, lambda1, lambda2, np.zeros(vertices), cluster_assignments, adjacency_matrix, degrees, transition_matrix, stoc, pattern)[6]

    if gtype == "facebook":
        ncps = np.zeros((vertices,))
        ncp_indices =  [0,107,348,414,686,698,1684,1912,3437,3980]
        ncps[ncp_indices] = 1
    else:
        ncps = generate_ncps(filename, graph, adjacency_matrix, transition_matrix, gtype=gtype, pattern="dominating", weight="influence", maximum=True)

    total_ncps = np.sum(ncps==1)

    frac_ncps_ary = []
    normalized_diff_ary = []
    if pattern == "dominating":
        for i in range(total_ncps):
            dom_ncps = np.zeros(len(graph.vs))
            dom_ncps[np.where(ncps==1)[0][0:i]] = 1
            (index, max_ncps, frac_uncovered, ncp_influence, 
                ate_without, ate_ncp_bias, ate_estimate_gui, beta, 
                gamma) = ate(graph, i, lambda0, lambda1, lambda2, dom_ncps, cluster_assignments, adjacency_matrix, degrees, transition_matrix, stoc, "dominating")

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
            (index, max_ncps, frac_uncovered, ncp_influence, 
                ate_without, ate_ncp_bias, ate_estimate_gui, beta, 
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
    for i in range(200,300):
        frac_ncps, normalized_diff = experiment(cwd+"/synthetic/"+code+"1000.net", gtype=code, lambda1=0.25, lambda2=0.0, pattern="dominating", weight="degree")
        plt.plot(frac_ncps, normalized_diff)
        # plt.show()

        f = open("/Users/kavery/workspace/correcting-ate-estimates/results/"+code+"_greedy_025_0/"+str(i)+".csv", 'w')
        writer = csv.writer(f)
        writer.writerow(frac_ncps)
        writer.writerow(normalized_diff)
        f.close()
    print("done")