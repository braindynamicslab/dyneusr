import numpy as np
import pandas as pd
import networkx as nx
from sklearn.datasets.base import Bunch


def make_random_graph(
        n_nodes=10, n_edges=10,
        n_features=10, n_local_samples=10, 
        random_state=None, verbose=1
    ):
    """Locally low-dimensional graph, embedded in high-dimensional space.

    Examples
    --------
    >>> from dyneusr.datasets import make_random_graph
    >>> dataset = make_random_graph(n_nodes=10, n_edges=10,
    ...                          n_features=100, n_local_samples=10, 
    ...                          random_state=0)
    >>> G, X, y = dataset.graph, dataset.data, dataset.target
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # sample random graph
    G = nx.random_regular_graph(2, n_nodes)
    X = np.random.random((n_nodes*n_local_samples, n_features))

    # empty list
    Z, Y = [], []    
    for n in G:
        for nbr in G.neighbors(n):
            # sample n_local_samples points between n, nbr
            line = pd.DataFrame(
                np.zeros((n_local_samples, n_features)),
                index=range(n_local_samples)
            )
            line.iloc[:, :] = np.nan
            line.iloc[0, :] = X[n].copy()
            line.iloc[-1, :] = X[nbr].copy()
            line = line.interpolate('linear')
            Z.append(line.copy().values)
            Y.append(np.ravel([n for _ in line]))

    # flatten Z
    X = np.array(Z).reshape(-1, n_features)
    y = np.ravel(Y)

    # return as bunch
    dataset = Bunch(
        graph=G,
        data=X,
        target=y,
    )
    return dataset
    
