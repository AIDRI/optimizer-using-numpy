from sgd_optimizer import SGD_no_momentum, SGD_momentum, SGD_nesterov
from adam_optimizer import Adam

func = lambda w : (1.5 - w[0] + w[0]*w[1])**2 + (2.25 - w[0] + w[0]*w[1]**2)**2 + (2.625 - w[0] + w[0]*w[1]**3)**2

opt = Adam(func)
#opt = SGD_no_momentum(func)
#opt = SGD_momentum(func)
#opt = SGD_nesterov(func)

opt.train()