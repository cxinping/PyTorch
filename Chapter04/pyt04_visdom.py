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
viz.image(np.ones((3, 10, 10)))








