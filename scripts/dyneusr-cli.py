import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from kmapper import KeplerMapper, Cover
from dyneusr import DyNeuGraph


class DyNeuSR(object):

    def init(self):
        return self

    def load_data(self, X=None, y=None): 
        """Load the data.

        Parameters
        ----------
        X : str
            Filename of data matrix to load.

        y : str, optional
            Filename of meta data to load.

        """
        # Helper functions
        def check_array_from_file(fp):
            print("Loading data from file:", fp)
            d = None
            if str(fp).endswith('.npy'):
                d = np.load(fp)
            elif str(fp).endswith('.npz'):
                d = np.loadz(fp)
                d = d[list(d.keys())[0]]
            elif str(fp).endswith('.tsv'):
                d = pd.read_table(fp)
            elif str(fp).endswith('.csv'):
                d = pd.read_csv(fp)
            elif str(fp).endswith('.txt'):
                d = np.genfromtxt(fp)
            else:
                print('Data format not recognized ...')
                print('Please use an accepted format:')
                print('\t.npy')
                print('\t.npz')
                print('\t.tsv')
                print('\t.csv')
                print('\t.txt')
            return d

        # Load the data from a file.
        X = check_array_from_file(X)
        y = check_array_from_file(y)
        dataset = Bunch(data=X, target=y)

        # Store as variables
        self.dataset = dataset
        self.X = X
        self.y = y
        return self


    def load_example(self, size=100):
        """Load the data.
        
        TODO
        ----
        - generalize to load any dataset supplied by the user

        """
        # Generate synthetic dataset (for now)
        from dyneusr.datasets import make_trefoil
        dataset = make_trefoil(size=size)
        X = dataset.data
        y = dataset.target

        # Store variables
        self.dataset = dataset
        self.X = X
        self.y = y 
        return self


    def run_mapper(self, 
               projection=[0],
               scaler=MinMaxScaler(),
               resolution=6, gain=0.2, 
               clusterer=KMeans(2),
               verbose=1):
        """Run KeplerMapper.
        """
        # Generate shape graph using KeplerMapper
        mapper = KeplerMapper(verbose=verbose)
        print(scaler)
        lens = mapper.fit_transform(
            self.X, 
            projection=projection, 
            scaler=scaler
        )
        graph = mapper.map(
            lens, self.X, 
            cover=Cover(resolution, gain), 
            clusterer=clusterer
        )

        # Store results
        self.lens = lens
        self.graph = graph
        return self


    def visualize(self, 
                  save_as='dyneusr_output.html',
                  template=None,
                  static=True, 
                  show=True, 
                  port=None):
        """Visualize the graph using DyNeuSR
        """
        # Visualize the shape graph using DyNeuSR's DyNeuGraph 
        dG = DyNeuGraph(G=self.graph, y=self.y)
        dG.visualize(
            save_as, 
            template=template,
            static=static,
            show=show,
            port=port
        )
        
        # Store the results
        self.dG = dG
        return self



if __name__=='__main__':
    import fire
    fire.Fire(DyNeuSR)
