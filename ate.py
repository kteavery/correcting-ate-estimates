import pandas as pd
import os
from igraph import Graph
import numpy as np
from sklearn.linear_model import LinearRegression

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

def ate(lambda1, lambda2, num_ncps, assignments, adjacency_matrix, degrees, transition_matrix):
    ate_without = lambda1+lambda2
    (non_ncp_treatment_neigh, 
        non_ncp_control_neigh, 
        ncp_treatment_neigh, 
        ncp_control_neigh, 
        adversarial_influence) = spillover_probs(num_ncps, assignments, adjacency_matrix, degrees, transition_matrix)
    # assignment + treatment_neighbors + ncp_treatment_neighbors + ncp_control_neighbors
    return ate_without

def check_set(adjacency_matrix, ncps): # check this? 
    adjacency_ncps = np.matmul(adjacency_matrix, np.transpose(ncps)) + np.transpose(ncps)
    return sum( np.sum(adjacency_ncps, axis=1) > 0 ) == np.shape(adjacency_matrix)[1]

def greedy(weight="influence"):
    return


def ncps(graph, adjacency_matrix, adversarial_influence, pattern="random", weight="influence", maximum=True):
    ncps = np.array([])
    if pattern == "random":
        if weight == "influence":
            if maximum:
                while not check_set(adjacency_matrix, ncps):
                    ncps[adversarial_influence[np.argsort(-1*adversarial_influence)]]
        else:
            if maximum:
                i = 1
                randlist = np.random.shuffle(np.array(range(0,11)))
                while not check_set(adjacency_matrix, ncps):
                    ncps[randlist[i]] = 1
                    i += 1

    else: # worst-case (dominating)
        # if weight == "influence":
        ncp_set = greedy(weight=weight)
        if maximum:
            num_ncps = len(ncp_set)
            ncps[np.random.shuffle(ncp_set)[:num_ncps]] = 1

    return ncps

def experiment(filename, noise=True, lambda1=0.75, lambda2=0.5, type="facebook"):
    graph = Graph.Read_Edgelist(filename, directed=False)
    degrees = Graph.degree(graph)
    vertices = len(graph.vs)
    adjacency_matrix = np.array(Graph.get_adjacency(graph).data)
    transition_matrix = np.matmul(np.diag(degrees), np.array(adjacency_matrix.data))

    # generate clusters
    clusters = graph.community_infomap()
    assignments = np.random.binomial(1, 0.5, len(clusters))

    # print(assignment)
    # ate
    ate(lambda1, lambda2, num_ncps, assignments, adjacency_matrix, degrees, transition_matrix)
    
    return


if __name__=='__main__':
    cwd = os.getcwd()
    experiment(cwd+"/synthetic/small500.net")
    # experiment(cwd+"/facebook/facebook_combined.net")