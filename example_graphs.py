from igraph import Graph
import os

def create_synthetic():
    cwd = os.getcwd()

    small500 = Graph.Watts_Strogatz(1, 20, 5, 0.05)
    small500.to_undirected()

    fire500 = Graph.Forest_Fire(n=20, fw_prob=0.32, bw_factor=0.33/0.32, directed=False)

    small500.save(cwd+"/synthetic/small20.net")
    fire500.save(cwd+"/synthetic/fire20.net")

if __name__=='__main__':
    create_synthetic()