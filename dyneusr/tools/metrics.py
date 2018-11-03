"""
Network metrics and analysis tools.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import networkx as nx

class Metric:

    def __init__(self, metric=None):
        self.metric = metric

    def evaluate(self, X, y=None):
        self.score_ = self.metric(X)
        return self.score_


class NetworkMetric(Metric):

    def __init__(self, metric=None):
        self.metric = metric

    def evaluate(self, G, y=None):
        self.score_ = self.metric(G)
        return self.score_


def degree(G):
    return nx.degree(G)

def modularity(G):
    pass

