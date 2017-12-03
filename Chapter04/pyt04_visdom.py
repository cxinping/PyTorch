# -*- coding: utf-8 -*-
 
"""
    【简介】
    PyTorch的AI可视化工具
        
"""


from visdom import Visdom
import numpy as np
import math
import os.path
import getpass
from sys import platform as _platform
from six.moves import urllib

viz = Visdom()

# 1
textwindow = viz.text('Hello Visdom !')

# 2 scatter plots
Y = np.random.rand(100)
viz.scatter(
    X=np.random.rand(100, 2),
    Y=(Y[Y > 0] + 1.5).astype(int),
    opts=dict(
        legend=['Apples', 'Pears'],
        xtickmin=-5,
        xtickmax=5,
        xtickstep=0.5,
        ytickmin=-5,
        ytickmax=5,
        ytickstep=0.5,
        markersymbol='cross-thin-open',
    ),
)






