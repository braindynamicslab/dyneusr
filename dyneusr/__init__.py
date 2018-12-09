try:
    import matplotlib as mpl
    mpl.use('TkAgg', warn=False)
except Exception as e:
    pass

__all__ = [
    'core', 
    'tools',
    'visuals',
    'datasets',
]

from .core import *
