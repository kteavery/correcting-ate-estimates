from igraph import Graph
import os

def create_synthetic():
    cwd = os.getcwd()

    small500 = Graph.Watts_Strogatz(1, 500, 5, 0.05)
    small1000 = Graph.Watts_Strogatz(1, 1000, 5, 0.05)
    small5000 = Graph.Watts_Strogatz(1, 5000, 5, 0.05)
    small500.to_undirected()
    small1000.to_undirected()
    small5000.to_undirected()

    fire500 = Graph.Forest_Fire(n=500, fw_prob=0.32, bw_factor=0.33/0.32, directed=False)
    fire1000 = Graph.Forest_Fire(n=1000, fw_prob=0.37, bw_factor=0.33/0.37, directed=False)
    fire5000 = Graph.Forest_Fire(n=5000, fw_prob=0.37, bw_factor=0.35/0.37, directed=False)

    small500.save(cwd+"/synthetic/small500.net")
    small1000.save(cwd+"/synthetic/small1000.net")
    small5000.save(cwd+"/synthetic/small5000.net")

    fire500.save(cwd+"/synthetic/fire500.net")
    fire1000.save(cwd+"/synthetic/fire1000.net")
    fire5000.save(cwd+"/synthetic/fire5000.net")

# def plot_graphs(filename):
#     graph = Graph.read_graph(filename)
#     plot(graph)

if __name__=='__main__':
    create_synthetic()
    # plot_graphs("/Users/kavery/workspace/correcting-ate-estimates/synthetic/fire500.net")