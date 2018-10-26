import numpy
import pandas
import matplotlib.pyplot as plt

def initialize(bandits):
	q = numpy.random.normal(0,1,bandits)
	Q = numpy.zeros(bandits)
	N = numpy.zeros(bandits)
	return q, Q, N

def activate(arm, q):
	return numpy.random.normal(q[arm],1)

def update(arm, Q ,reward, N):
	Q[arm] = Q[arm] + (reward-Q[arm])/N[arm]
	return Q

def train(q,Q,N,e, epochs):
	scores = []
	score = 0
	best = q.argmax()
	for i in range(epochs):
		if numpy.random.random() <= e:
			arm = numpy.random.choice(range(len(q)))
		else:
			ocurrences = numpy.where(Q == Q.max())[0]
			arm = numpy.random.choice(ocurrences)
		if best == arm:
			score += 1
		scores.append(score/(i+1))
		N[arm] += 1
		Rt = activate(arm,q)
		Q = update(arm, Q, Rt, N)
	return scores

def multiple_tests(n_tests, bandits, e, epochs):
	scores_list = []
	for i in range(n_tests):
		q,Q,N = initialize(bandits)
		scores_list.append(train(q,Q,N,e,epochs))
	return scores_list

def perform_and_plot_multiple_tests(es, n_tests, bandits, epochs):
	results = {}
	for e in es:
		results[e] = pandas.DataFrame(multiple_tests(n_tests, bandits, e, epochs)).mean()
	for e in results:
		plt.plot(results[e], label = e)
	plt.legend()
	plt.show()
