class Adam():
	def __init__(self, func, partial_derivative):
		self.w = [2.5, 1.2] #random value
		self.mean = [0, 0]
		self.variance = [0, 0]
		self.unbiased_mean = [0, 0]
		self.unbiased_variance = [0, 0]
		self.n = 0.001
		self.e = 1e-7
		self.func = func
		self.partial_derivative = partial_derivative

	def theta_update(self, grads):
		self.mean[0] = (1 - 0.9) * grads[0] + 0.9 * self.mean[0]
		self.mean[1] = (1 - 0.9) * grads[1] + 0.9 * self.mean[1]

		self.variance[0] = (1 - 0.98) * grads[0] ** 2 + 0.98 * self.variance[0]
		self.variance[1] = (1 - 0.98) * grads[1] ** 2 + 0.98 * self.variance[1]

		self.unbiased_mean[0] = self.mean[0] / (1 - 0.9 ** 2)
		self.unbiased_mean[1] = self.mean[1] / (1 - 0.9 ** 2)

		self.unbiased_variance[0] = self.variance[0] / (1 - 0.98 ** 2)
		self.unbiased_variance[1] = self.variance[1] / (1 - 0.98 ** 2)

		self.w[0] -= (self.n * self.unbiased_mean[0]) / ((self.unbiased_variance[0])**0.5 + self.e)
		self.w[1] -= (self.n * self.unbiased_mean[1]) / ((self.unbiased_variance[1])**0.5 + self.e)

	def train(self):
		J_min = 0
		J = self.func(self.w)
		epochs = 0
		print(J_min, J)
		while abs(J_min - J) > 1e-8:
			if epochs != 0:
				J_min = J
			J = self.func(self.w)
			self.grad = self.partial_derivative(self.w)
			self.theta_update(self.grad)
			epochs += 1
			if epochs%500==0:
				print('Epochs : {} ; Cost : {} ; X : {} ; Y : {}'.format(epochs, J, self.w[0], self.w[1]))

		print('Epochs : {} ; Cost : {} ; X : {} ; Y : {}'.format(epochs, J, self.w[0], self.w[1]))
		print('Global Minima : {} ; {}'.format(self.w[0], self.w[1]))
