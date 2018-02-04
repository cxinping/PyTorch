from visdom import Visdom
import numpy as np
import torch
vis = Visdom()

vis.scatter(
    X =  torch.rand(255, 2),
    Y = (torch.randn(255) > 0) + 1 ,
       opts=dict(
        markersize=10,
        legend=['Men', 'Women']
    ),
)
