try:
    import matplotlib as mpl
    mpl.use('TkAgg')
except Exception as e:
    pass

__all__ = [
    'core', 
    'tools',
    'visuals'
]

from .core import *
