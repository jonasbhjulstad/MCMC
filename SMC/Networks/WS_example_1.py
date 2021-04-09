import networkx as nx
import networkx as nx
import matplotlib.pyplot as plt
if __name__ == '__main__':


    G = nx.watts_strogatz_graph(n=10, k=4, p=1)
    pos = nx.circular_layout(G)

    plt.figure(figsize=(12, 12))
    nx.draw_networkx(G, pos)

    plt.show()