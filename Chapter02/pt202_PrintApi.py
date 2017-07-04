# -*- coding: utf-8 -*- 
'''
    【简介】
	 
    
'''

import sys
import torch

out = sys.stdout
sys.stdout = open(r'./log.txt' , 'w')
help( torch.Tensor  )
sys.stdout.close()
sys.stdout = out
