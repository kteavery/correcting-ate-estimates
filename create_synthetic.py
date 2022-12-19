from igraph import Graph
import os
import numpy as np

def create_synthetic():
    cwd = os.getcwd()
    
    small1000 = Graph.Watts_Strogatz(1, 1000, 5, 0.05)
    small1000.to_undirected()
    
    edges = 4697
    count = 0
    while count < (edges-edges*.1) or count > (edges+edges*.1):
        fire1000 = Graph.Forest_Fire(n=1000, fw_prob=0.37, bw_factor=0.33/0.37, directed=False)
        adj = np.array( Graph.get_adjacency(fire1000).data )
        count = np.sum(adj)/2

    small1000.save(cwd+"/synthetic/small1000.net")
    fire1000.save(cwd+"/synthetic/fire1000.net")


if __name__=='__main__':
    create_synthetic()