from igraph import Graph
import os
import numpy as np

def create_synthetic():
    cwd = os.getcwd()
    
    small500 = Graph.Watts_Strogatz(1, 500, 3, 0.05)
    small1000 = Graph.Watts_Strogatz(1, 1000, 5, 0.05)
    small5000 = Graph.Watts_Strogatz(1, 5000, 25, 0.05)
    small500.to_undirected()
    small1000.to_undirected()
    small5000.to_undirected()

    edges = 1370
    # edges = 4697
    count = 0
    while count < (edges-edges*.1) or count > (edges+edges*.1):
        # fire500 = Graph.Forest_Fire(n=500, fw_prob=0.32, bw_factor=0.33/0.32, directed=False)
        fire500 = Graph.Forest_Fire(n=500, fw_prob=0.37, bw_factor=0.33/0.37, directed=False)
        # fire5000 = Graph.Forest_Fire(n=5000, fw_prob=0.37, bw_factor=0.35/0.37, directed=False)
        adj = np.array( Graph.get_adjacency(fire500).data )
        count = np.sum(adj)/2

    # small500.save(cwd+"/synthetic/small500.net")
    # small1000.save(cwd+"/synthetic/small1000.net")
    # small5000.save(cwd+"/synthetic/small5000.net")

    # fire500.save(cwd+"/synthetic/fire500.net")
    fire500.save(cwd+"/synthetic/fire500.net")
    fire500.save(cwd+"/synthetic/fire500.net")
    fire5000.save(cwd+"/synthetic/fire5000.net")



if __name__=='__main__':
    create_synthetic()
    # plot_graphs("/Users/kavery/workspace/correcting-ate-estimates/synthetic/fire500.net")