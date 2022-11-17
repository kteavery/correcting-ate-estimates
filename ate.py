import pandas as pd
import os
from igraph import Graph
import numpy as np
from sklearn.linear_model import LinearRegression

def spillover_probs():
    return

def ate(lambda1, lambda2):
    ate_without = lambda1+lambda2
    # assignment + treatment_neighbors + ncp_treatment_neighbors + ncp_control_neighbors
    return ate_without

def ncps(graph):
    return

def experiment(filename, noise=True, lambda1=0.75, lambda2=0.5):
    # df = pd.read_csv(filename)
    graph = Graph.Read_Edgelist(filename, directed=False)
    # generate clusters
    clusters = graph.community_infomap()
    assignment = np.random.binomial(1, 0.5, len(clusters))
    # print(assignment)
    # ate
    ate(lambda1, lambda2)
    
    return


if __name__=='__main__':
    cwd = os.getcwd()
    experiment(cwd+"/synthetic/small500.net")
    # experiment(cwd+"/facebook/facebook_combined.net")