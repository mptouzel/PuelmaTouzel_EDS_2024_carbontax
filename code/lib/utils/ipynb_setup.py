# computation
from IPython.display import Math
from IPython.display import display, HTML
from importlib import reload
import itertools
from functools import partial
from copy import deepcopy
import time
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from itertools import product
from scipy import special
from shutil import which

# plotting
import matplotlib.pyplot as pl
import seaborn as sns
sns.set_style("ticks", {'axes.grid': True})
pl.rc("figure", facecolor="white", figsize=(8, 8))
#pl.rc("figure", facecolor="gray",figsize = (8,8))
if which('latex'):
    pl.rc('text', usetex=True)
    pl.rc('text.latex', preamble=r'\usepackage{amsmath}')
pl.rc('lines', markeredgewidth=2)
pl.rc('font', size=10)

# notebook config
display(HTML("<style>.container { width:100% !important; }</style>"))
