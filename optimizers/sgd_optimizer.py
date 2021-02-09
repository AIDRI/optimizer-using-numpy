class SGD_no_momentum():
	def __init__(self, func, derivative):
		self.w = [-10, -1] #random value
		self.e = 0.0005
		self.func = func
		self.derivative = derivative

	def theta_update(self, w):
		self.w[0] -= self.e * self.grad[0]
		self.w[1] -= self.e * self.grad[1]

	def train(self):
		J_min = 0
		J = self.func(self.w)
		epochs = 0
		print(J_min, J)
		while abs(J_min - J) > 1e-8:
			if epochs != 0:
				J_min = J
			J = self.func(self.w)
			self.grad = self.derivative(self.w)
			self.theta_update(self.grad)
			epochs += 1
			if epochs%10==0:
				print('Epochs : {} ; Cost : {}'.format(epochs, J))

		print('Epochs : {} ; Cost : {}'.format(epochs, J))
		print('Global Minima : {} ; {}'.format(self.w[0], self.w[1]))


class SGD_momentum():
	def __init__(self, func, derivative):
		self.w = [2.5, 1.2] #random value
		self.v = [0, 0]
		self.e = 0.0001
		self.func = func
		self.derivative = derivative

	def theta_update(self, grads):
		self.v[0] = 0.9 * self.v[0] - self.e * grads[0]
		self.w[0] += self.v[0]
		self.v[1] = 0.9 * self.v[1] - self.e * grads[1]
		self.w[1] += self.v[1]

	def train(self):
		J_min = 0
		J = self.func(self.w)
		epochs = 0
		print(J_min, J)
		while abs(J_min - J) > 1e-8:
			if epochs != 0:
				J_min = J
			J = self.func(self.w)
			self.grad = self.derivative(self.w)
			self.theta_update(self.grad)
			epochs += 1
			if epochs%500==0:
				print('Epochs : {} ; Cost : {} ; X : {} ; Y : {}'.format(epochs, J, self.w[0], self.w[1]))

		print('Epochs : {} ; Cost : {} ; X : {} ; Y : {}'.format(epochs, J, self.w[0], self.w[1]))
		print('Global Minima : {} ; {}'.format(self.w[0], self.w[1]))


class SGD_nesterov():
	def __init__(self, func, derivative):
		self.w = [2.5, 1.2] #random value
		self.v = [0, 0]
		self.e = 0.0001
		self.func = func
		self.derivative = derivative

	def theta_update(self, grads):
		self.v[0] = 0.9 * self.v[0] - self.e * grads[0]
		self.w[0] += self.v[0]
		self.v[1] = 0.9 * self.v[1] - self.e * grads[1]
		self.w[1] += self.v[1]

	def train(self):
		J_min = 0
		J = self.func(self.w)
		epochs = 0
		print(J_min, J)
		while abs(J_min - J) > 1e-8:
			if epochs != 0:
				J_min = J
			J = self.func(self.w)
			X_ = self.w[0] + 0.9 * self.v[0]
			Y_ = self.w[1] + 0.9 * self.v[1]
			W_ = [X_, Y_]
			self.grad = self.derivative(W_)
			self.theta_update(self.grad)
			epochs += 1
			if epochs%500==0:
				print('Epochs : {} ; Cost : {} ; X : {} ; Y : {}'.format(epochs, J, self.w[0], self.w[1]))

		print('Epochs : {} ; Cost : {} ; X : {} ; Y : {}'.format(epochs, J, self.w[0], self.w[1]))
		print('Global Minima : {} ; {}'.format(self.w[0], self.w[1]))
