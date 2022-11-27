import pandas as pd
import os
from igraph import Graph
import numpy as np
from sklearn.linear_model import LinearRegression
import copy

def spillover_probs(num_ncps, assignments, adjacency_matrix, degrees, transition_matrix):
    non_ncp_treatment = assignments * (1-num_ncps)
    non_ncp_treatment_neigh = np.transpose(np.matmul(adjacency_matrix, np.transpose(non_ncp_treatment) / degrees))
    
    non_ncp_control = (1-assignments) * (1-num_ncps)
    non_ncp_control_neigh = np.transpose(np.matmul(adjacency_matrix, np.transpose(non_ncp_control) / degrees))
    
    ncp_treatment = assignments * num_ncps
    ncp_treatment_neigh = np.transpose(np.matmul(adjacency_matrix, np.transpose(ncp_treatment) / degrees))
    
    ncp_control = num_ncps - treatment_ncps
    ncp_control_neigh = np.transpose(np.matmul(adjacency_matrix, np.transpose(ncp_control) / degrees))
    
    adversarial_influence = np.sum(transition_matrix, axis=0)
    return non_ncp_treatment_neigh, non_ncp_control_neigh, ncp_treatment_neigh, ncp_control_neigh, adversarial_influence

def outcome(lambda0, lambda_1, lambda_2, assignments, adjacency_matrix, degrees, ncps):
    outcome = np.array([])
    return outcome

def linear(adjacency_matrix, degrees, assignments, y):
    amt_treated = np.matmul(adjacency_matrix, assignments) / degrees
    lm = LinearRegression()
    return lm.fit(assignments + amt_treated, y)

def linear_ncp(non_ncp_treatment_neigh, ncp_treatment_neigh, ncp_control_neigh, assignments, y):
    lm = LinearRegression()
    return lm.fit(assignments + amt_treated, y)

#function(idx, graph.properties, adversaries, outcome.params, ncp.params, treatment.assignments, stochastic.vars, bias.behavior)
def ate(lambda1, lambda2, num_ncps, cluster_assignments, adjacency_matrix, degrees, transition_matrix, stochastic):
    ate_without = lambda1+lambda2
    (non_ncp_treatment_neigh, 
        non_ncp_control_neigh, 
        ncp_treatment_neigh, 
        ncp_control_neigh, 
        adversarial_influence) = spillover_probs(num_ncps, assignments, adjacency_matrix, degrees, transition_matrix)
    
    return index, dominate_size, method, pt_uncovered, ncp_inf, ate_true, ate_ncp_estimate, ate_ncp_gui, gui_beta, gui_gamma

def check_set(adjacency_matrix, ncps): # check this? 
    adjacency_ncps = np.matmul(adjacency_matrix, np.transpose(ncps)) + np.transpose(ncps)
    return sum( np.sum(adjacency_ncps, axis=1) > 0 ) == np.shape(adjacency_matrix)[1]

def greedy(filename, graph, adjacency_matrix, transition_matrix, weight="influence"):
    if weight == "influence":
        trans_sum = np.sum(transition_matrix, axis=0)
    else:
        dir_graph = Graph.Read_Edgelist(filename, directed=False)
        trans_sum = Graph.outdegree(graph)
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
        
        greedy = np.append(greedy, argmax)
        if weight == "influence":
            for i in greedy:
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


def generate_ncps(filename, graph, adjacency_matrix, transition_matrix, pattern="random", weight="influence", maximum=True):
    ncps = np.zeros(len(graph.vs)-1)
    if pattern == "random":
        if weight == "influence":
            if maximum:
                while not check_set(adjacency_matrix, ncps):
                    ncps[transition_matrix[np.argsort(-1*transition_matrix)]]
        else:
            if maximum:
                i = 1
                randlist = np.random.shuffle(np.array(range(0,11)))
                while not check_set(adjacency_matrix, ncps):
                    ncps[randlist[i]] = 1
                    i += 1

    else: # worst-case (dominating)
        ncp_set = greedy(filename, graph, adjacency_matrix, transition_matrix, weight=weight)
        print(ncp_set)
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

def experiment(filename, lambda1=0.75, lambda2=0.5, pattern="random", gtype="facebook"):
    graph = Graph.Read_Edgelist(filename, directed=False)
    degrees = Graph.degree(graph)
    vertices = len(graph.vs)
    adjacency_matrix = np.array(Graph.get_adjacency(graph).data)
    transition_matrix = np.matmul(np.diag(degrees), np.array(adjacency_matrix.data))

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
        stoc = stochastic(vertices-1, 3, 0.1)
    # print(stoc)
    # ate
    #0, graph.properties, matrix(0,1,graph.properties$n), outcome.params, ncp.params, treatment.assignments, stochastic.vars, bias.behavior)$ATE.adv.gui[1]
    # non_ncp_ate = ate(lambda1, lambda2, num_ncps, cluster_assignments, adjacency_matrix, degrees, transition_matrix, stoc)
    if gtype == "facebook":
        ncps = np.zeros((vertices-1,))
        ncp_indices = [1,108,349,415,687,699,1685,1913,3438,3981]
        ncps = np.array([ncps[i] for i in ncp_indices])
    else:
        ncps = generate_ncps(filename, graph, adjacency_matrix, transition_matrix, pattern="dominating", weight="degree", maximum=True)
    print(ncps)

    # if pattern == "random":
    #     for i in 
    #     ncp_ate = ate(lambda1, lambda2, num_ncps, cluster_assignments, adjacency_matrix, degrees, transition_matrix, stoc)
    # else: # dominating
    #     if gtype == "facebook":
    #         ncp_ate = ate(lambda1, lambda2, num_ncps, cluster_assignments, adjacency_matrix, degrees, transition_matrix, stoc)
    #     else: 
    #         ncp_ate = ate(lambda1, lambda2, num_ncps, cluster_assignments, adjacency_matrix, degrees, transition_matrix, stoc)


    return


if __name__=='__main__':
    cwd = os.getcwd()
    experiment(cwd+"/synthetic/small500.net", gtype="small")
    # experiment(cwd+"/facebook/facebook_combined.net")