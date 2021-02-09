from optimizers.sgd_optimizer import SGD_no_momentum, SGD_momentum, SGD_nesterov
from optimizers.adam_optimizer import Adam

func = lambda w : (1.5 - w[0] + w[0]*w[1])**2 + (2.25 - w[0] + w[0]*w[1]**2)**2 + (2.625 - w[0] + w[0]*w[1]**3)**2

def partial_derivative(w):
		x, y = w[0], w[1]
		dx = 2 * ((1.5-x+x*y) * (y-1)+(2.25-x+x*y**2) * (y**2-1)+(2.625-x+x*y**3)*(y**3-1))
		dy = 2 * ((1.5-x+x*y) * x+(2.25-x+x*y**2) * 2*x*y+(2.625-x+x*y**3) * 3*x*y**2)
		return [dx, dy]


opt = Adam(func, partial_derivative)
#opt = SGD_no_momentum(func)
#opt = SGD_momentum(func)
#opt = SGD_nesterov(func)

opt.train()
