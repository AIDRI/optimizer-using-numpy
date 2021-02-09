import numpy as np
from sgd_optimizer import SGD_no_momentum, SGD_momentum, SGD_nesterov


func = lambda w : (1.5 - w[0] + w[0]*w[1])**2 + (2.25 - w[0] + w[0]*w[1]**2)**2 + (2.625 - w[0] + w[0]*w[1]**3)**2

opt = SGD_nesterov(func)
opt.train()