# -*- coding: utf-8 -*-
 
"""
    【简介】
    PyTorch的AI可视化工具
        
"""

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#from __future__ import unicode_literals

from visdom import Visdom
import numpy as np
import math
import os.path
import getpass
from sys import platform as _platform
from six.moves import urllib

viz = Visdom()
textwindow = viz.text('Hello Visdom.')









