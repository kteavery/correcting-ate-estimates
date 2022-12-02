import pandas as pd
import os
from igraph import Graph
import numpy as np
from sklearn.linear_model import LinearRegression
import copy
import warnings
warnings.filterwarnings("ignore")

np.random.seed(1)

def spillover_probs(ncps, assignments, adjacency_matrix, degrees, transition_matrix):
    non_ncp_treatment = assignments * (1-ncps)
    non_ncp_treatment_trans = np.transpose(non_ncp_treatment) / degrees
    non_ncp_treatment_trans[np.isnan(non_ncp_treatment_trans)] = 0
    # print(non_ncp_treatment_trans)
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
    # print("(non)ncps")
    # print(non_ncp_treatment, non_ncp_control, ncp_treatment, ncp_control)
    # print("neighbors")
    # print(non_ncp_treatment_neigh, non_ncp_control_neigh, ncp_treatment_neigh, ncp_control_neigh, adversarial_influence)
    return non_ncp_treatment_neigh, non_ncp_control_neigh, ncp_treatment_neigh, ncp_control_neigh, adversarial_influence

def outcome(graph, lambda0, lambda1, lambda2, cluster_assignments, adjacency_matrix, degrees, ncps, stochastic):
    outcome = np.zeros(len(graph.vs))
    # print(outcome)
    treatment_ncps = cluster_assignments * ncps
    control_ncps = ncps - treatment_ncps

    for i in range(0,3):
        coef2 = np.matmul(adjacency_matrix, np.diag(outcome))/degrees
        coef2[np.isnan(coef2)] = 0
        outcome = lambda0 + lambda1*cluster_assignments + lambda2*np.sum(coef2, axis=1) + stochastic[:,i]
        intermediate = np.zeros(len(treatment_ncps))
        intermediate[np.where(treatment_ncps==1)] = lambda0
        intermediate[np.where(control_ncps==1)] = lambda0+lambda1
        intermediate = intermediate[np.where(ncps==1)]
        # print(outcome.shape)
        # print(intermediate.shape)
        # print(outcome)
        # print(intermediate)
        outcome[np.where(ncps==1)] = intermediate
    return outcome

def linear(adjacency_matrix, degrees, assignments, outcome):
    amt_treated = np.matmul(adjacency_matrix, assignments) / degrees
    amt_treated[np.isnan(amt_treated)] = 0.0
    lm = LinearRegression()
    X = np.array([assignments, amt_treated])
    return lm.fit(np.transpose(X), outcome)

def linear_ncp(non_ncp_treatment_neigh, ncp_treatment_neigh, ncp_control_neigh, assignments, outcome):
    lm = LinearRegression()
    # print(outcome.shape)
    # print(assignments)
    # print(non_ncp_treatment_neigh)
    # print(ncp_treatment_neigh)
    # print(ncp_control_neigh)
    X = np.array([assignments, non_ncp_treatment_neigh, ncp_treatment_neigh, ncp_control_neigh])
    # X = X.reshape(-1, 1)
    # print(X.shape)
    return lm.fit(np.transpose(X), outcome)

#function(idx, graph.properties, adversaries, outcome.params, ncp.params, treatment.assignments, stochastic.vars, bias.behavior)
def ate(graph, index, lambda0, lambda1, lambda2, ncps, cluster_assignments, adjacency_matrix, degrees, transition_matrix, stochastic, pattern="random"):
    ate_without = lambda1+lambda2
    # print(ncps.shape)
    # print(adjacency_matrix.shape)
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
    # print(lin_ncp.coef_)
    # print(lin_ncp.intercept_)
    # print(lin_ncp)

    lin_gui = linear(adjacency_matrix, degrees, cluster_assignments, ncp_outcome)
    # print(lin_gui.coef_)
    beta = lin_gui.coef_[0]
    gamma = lin_gui.coef_[1]
    ate_estimate_gui = beta+gamma
    # print(lin_gui)
    # ate_gui = lin_gui

    if pattern == "dominating" or index == 0:
        max_ncps = False
    else:
        max_ncps = True
    
    ncp_influence = np.sum(adversarial_influence[np.where(ncps==1)])/(len(graph.vs))

    return index, max_ncps, frac_uncovered, ncp_influence, ate_without, ate_ncp_bias, ate_estimate_gui, beta, gamma

def check_set(adjacency_matrix, ncps): # check this? 
    adjacency_ncps = np.matmul(adjacency_matrix, np.transpose(ncps)) + np.transpose(ncps)
    # print(np.sum(adjacency_ncps) > 0)
    # print("adjacency_ncps")
    # print(np.sum(adjacency_ncps, axis=0))
    # print(adjacency_ncps)
    print((adjacency_ncps > 0 ).sum())
    print((adjacency_ncps > 0 ).sum() == np.shape(adjacency_matrix)[1])
    return (adjacency_ncps > 0 ).sum() == np.shape(adjacency_matrix)[1]

def greedy(filename, graph, adjacency_matrix, weight="influence"):
    dir_graph = Graph.Read_Edgelist(filename, directed=False)
    adjacency_matrix = np.array(Graph.get_adjacency(dir_graph).data) 
    if weight == "influence":
        degrees = Graph.outdegree(dir_graph)
        inverse = np.diag(np.reciprocal( np.array(degrees).astype(float) ))
        inverse[np.isinf(inverse)] = 0
        transition_matrix = np.matmul(inverse, np.array(adjacency_matrix.data))
        trans_sum = np.sum(transition_matrix, axis=0)
    else:
        dir_graph = Graph.Read_Edgelist(filename, directed=False)
        trans_sum = Graph.outdegree(dir_graph)
    # print("trans_sum")
    # print(transition_matrix)
    # print(trans_sum)
    adjacency = copy.deepcopy(adjacency_matrix)
    cover = np.zeros(len(adjacency))
    greedy = np.array([])
    while np.sum(cover) < len(adjacency):
        argmaxs = np.where(trans_sum==np.max(trans_sum)) 
        argmax = int(np.random.choice(argmaxs[0], 1))
        cover[adjacency[argmax]>0] = 1
        cover[argmax] = 1
        # print(argmaxs)
        # print(argmax)
        # print(adjacency[argmax]>0)
        # print(cover)
        
        greedy = np.append(greedy, argmax)
        # print("greedy")
        # print(greedy)
        if weight == "influence":
            trans_matrix_temp = np.matmul(inverse, np.array(adjacency_matrix.data))
            # print(trans_matrix_temp.shape)
            # print("greedy")
            # print(greedy)
            for i in greedy.astype(int):
                # print(trans_matrix_temp[i])
                trans_matrix_temp[i,:] = 0
                trans_matrix_temp[:,i] = 0
            trans_sum = np.sum(trans_matrix_temp, axis=0)
        else: 
            directed = np.array(Graph.get_adjacency(dir_graph).data) 
            directed[cover>0]=0 
            dir_graph = Graph.Adjacency(directed)
            trans_sum = Graph.outdegree(dir_graph)-cover
        # print(cover)
        # print(np.sum(cover))
        # print(len(adjacency))
    return greedy 


def generate_ncps(filename, graph, adjacency_matrix, transition_matrix, pattern, weight, maximum):
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
                # print("rand_list")
                # print(rand_list)
                while not check_set(adjacency_matrix, ncps):
                    ncps[randlist[i]] = 1
                    i += 1

    else: # worst-case (greedy)
        ncp_set = greedy(filename, graph, adjacency_matrix, weight=weight)
        ncp_set = np.unique(ncp_set)
        print("ncp_set")
        print(len(ncp_set))
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

def experiment(filename, pattern, weight, gtype, lambda0=-1.5, lambda1=0.75, lambda2=0.5):
    graph = Graph.Read_Edgelist(filename, directed=False)
    degrees = Graph.degree(graph)
    vertices = len(graph.vs)
    adjacency_matrix = np.array(Graph.get_adjacency(graph).data)
    inverse = np.diag(np.reciprocal( np.array(degrees).astype(float) ))
    inverse[np.isinf(inverse)] = 0
    transition_matrix = np.matmul(inverse, np.array(adjacency_matrix.data))
    # print("adj")
    # print(np.array(adjacency_matrix.data))
    # print("degrees")
    # print(np.diag(degrees).astype(float))
    # print(inverse)
    # print("transition")
    # print(transition_matrix)

    # generate clusters
    clusters = graph.community_infomap()
    assignments = np.random.binomial(1, 0.5, len(clusters))
    cluster_assignments = np.array([])
    for i in range(len(clusters)):
        curr_cluster = np.array(clusters[i])
        curr_cluster.fill(assignments[i])
        cluster_assignments = np.append(cluster_assignments, curr_cluster)
    # print(cluster_assignments)
    # print(assignments)

    if gtype == "facebook":
        stoc = None
    else: 
        stoc = stochastic(vertices, 3, 0.1)
    # print(stoc)
    # ate
    #0, graph.properties, matrix(0,1,graph.properties$n), outcome.params, ncp.params, treatment.assignments, stochastic.vars, bias.behavior)$ATE.adv.gui[1]
    # ate(graph, index, lambda1, lambda2, ncps, cluster_assignments, adjacency_matrix, degrees, transition_matrix, stochastic, pattern="random"):
    non_ncp_ate = ate(graph, 0, lambda0, lambda1, lambda2, np.zeros(vertices), cluster_assignments, adjacency_matrix, degrees, transition_matrix, stoc, pattern)[6]
    # print(non_ncp_ate)
    if gtype == "facebook":
        ncps = np.zeros((vertices-1,))
        ncp_indices = [1,108,349,415,687,699,1685,1913,3438,3981]
        ncps = np.array([ncps[i] for i in ncp_indices])
    else:
        ncps = generate_ncps(filename, graph, adjacency_matrix, transition_matrix, pattern=pattern, weight=weight, maximum=True)
    # print("ncps")
    # print(ncps)
    total_ncps = np.sum(ncps==1)

    if pattern == "dominating":
        for i in range(total_ncps):
            dom_ncps = np.zeros(len(graph.vs))
            dom_ncps[np.where(ncps==1)[0][0:i]] = 1
            # print("np.sum(dom_ncps)")
            # print(i)
            # print(np.where(ncps==1)[0][0:i])
            # print(np.sum(dom_ncps))
            (index, max_ncps, frac_uncovered, ncp_influence, 
                ate_without, ate_ncp_bias, ate_estimate_gui, beta, 
                gamma) = ate(graph, i, lambda0, lambda1, lambda2, dom_ncps, cluster_assignments, adjacency_matrix, degrees, transition_matrix, stoc, "dominating")
            # ncps = dom_ncps

            frac_ncps = index/len(graph.vs)
            difference = non_ncp_ate - ate_estimate_gui
            normalized_diff = difference/non_ncp_ate
            if frac_ncps < 0.3:
                print()
                print(frac_ncps)
                print(normalized_diff)

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
            if frac_ncps < 0.3:
                print()
                print(frac_ncps)
                print(normalized_diff)

    return index, frac_uncovered, ncp_influence, ate_without, non_ncp_ate, ate_ncp_bias, ate_estimate_gui, beta, gamma


if __name__=='__main__':
    cwd = os.getcwd()
    (index, frac_uncovered, ncp_influence, ate_without, non_ncp_ate,
        ate_ncp_bias, ate_estimate_gui, beta, 
        gamma) = experiment(cwd+"/synthetic/fire1000.net", gtype="fire", lambda1=0.25, lambda2=1.0, pattern="dominating", weight="influence")
    # experiment(cwd+"/facebook/facebook_combined.net")

    # frac_ncps = index/500
    # difference = non_ncp_ate - ate_estimate_gui
    # normalized_diff = difference/non_ncp_ate

    # print(frac_ncps)
    # print(normalized_diff)