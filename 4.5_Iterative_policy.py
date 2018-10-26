import numpy
from scipy.stats.distributions import poisson
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

def policy_evaluation(V, pi, probabilitiesx, probabilitiesy,gamma = 0.9, sigma = 1, max_lambda = 12):
	"""
	Computes the policy evaluation step. It takes a as parameters:
	-V :set of arbitrary (or previous) values-state.
	-pi : arbitrary (or previous) policy
	- probabilitiesx and probabilitiesy: set of pre-computed poission distributed probabiblities for
	each parking lot. They are precomputed because doing it more than necessary uses a lot of resources.
	- gamma: the discounting factor to ensure convergence.
	- sigma: the maximum difference between any pair of new-old state to accept convergence.
	- max_lambda: the ammount of maximum value to expect from the poission distribution. To avoid unnecessary
	calculations. Plainly speaking: it is higly improbable that with a k = 3 we could reach a 10. So we 
	can avoid considering them.
	"""
	while True:
		delta = 0
		i = 0
		for statex in range(0,21):
			for statey in range(0,21):
				i+= 1
				print('\r' + str(round(i/441*100,3)) + '% of policy evaluation', end = '')
				action = pi[statex, statey]
				V0 = V[statex, statey]
				V[statex, statey] = compute_bellman_update(statex,statey,action,probabilitiesx, probabilitiesy,max_lambda, gamma)
				delta = max(delta, abs(V[statex, statey] - V0))
		print('\n delta score is: ',delta)
		if delta > sigma:
			print('Delta is not small enought, repeating the policy evaluation')
			continue
		else:
			return V

def compute_probabiblities_all(picklambdax = 3, picklambday = 4, returnlambdax = 3, returnlambday = 2):
	probabilitiesx = numpy.zeros([21, 21])
	probabilitiesy = numpy.zeros([21, 21])
	for picks in range(0, 21):
		for returns in range(0, 21):
			probabilitiesx[picks, returns] += round(poisson.pmf(picks, picklambdax)*poisson.pmf(returns, returnlambdax), 30)
			probabilitiesy[picks, returns] += round(poisson.pmf(picks, picklambday)*poisson.pmf(returns, returnlambday), 30)
	return probabilitiesx, probabilitiesy

def compute_reward(statex, statey, nclientsx, nclientsy, action):
	rentalsx = min(nclientsx, statex)
	rentalsy = min(nclientsy, statey)
	reward = 10*(rentalsx + rentalsy)
	if action != 0:
		reward -=  2 * (abs(action) - 1)
	if statex > 10:
		reward -= 4
	if statey > 10:
		reward -= 4
	return reward

def compute_bellman_update(statex,statey,action,probabilitiesx, probabilitiesy,max_lambda, gamma):
	v = 0
	statex += action
	statey -= action
	for picksx in range(min(max_lambda,statex+1)):
		for picksy in range(min(max_lambda,statey+1)):
			for returnsx in range(max_lambda):
				for returnsy in range(max_lambda):
					statex1 = min(max((statex+picksx), 20)-returnsx,0)
					statey1 = min(max((statey+picksy), 20)-returnsy, 0)
					reward = compute_reward(statex, statey, picksx, picksy, action) 
					v += probabilitiesx[picksx, returnsy]*probabilitiesy[picksy, returnsy] * (reward + gamma * V[statex1, statey1])
	return v

def initialization(states_shape, n_actions):
	V = numpy.zeros(states_shape)
	pi = numpy.zeros(states_shape,dtype=numpy.int8)
	A = numpy.arange(-5, 6)
	return V, pi,A

def policy_improvement(V, pi,A, probabilitiesx, probabilitiesy, max_lambda, gamma):
	policy_stable = True
	i = 0
	for statex in range(21):
		for statey in range(21):
			i+= 1
			print('\r' + str(round(i/441*100,3)) + '% of policy improvement', end = '')
			old_action = pi[statex, statey]
			pi[statex, statey] = compute_maxaV(V,A,statex, statey,probabilitiesx, probabilitiesy, max_lambda, gamma)
			if old_action != pi[statex, statey]:
				policy_stable = False
	if policy_stable:
		print('Policy improvement finished. The policy is stable')
	else:
		print('Policy improvement finished. The policy is NOT stable')
	return pi, policy_stable

def compute_maxaV(V,A, statex, statey,probabilitiesx, probabilitiesy, max_lambda, gamma):
	max_a = 0
	max_q = 0
	for action in A:
		if statex + action < 0:
			continue
		elif statey - action < 0:
			continue
		qva = compute_bellman_update(statex, statey, action, probabilitiesx, probabilitiesy, max_lambda, gamma) 
		if qva > max_q:
			max_q = qva
			max_a = action
	return max_a


V, pi, A = initialization([21,21],21)
probabilitiesx, probabilitiesy = compute_probabiblities_all()
V = policy_evaluation(V, pi, probabilitiesx, probabilitiesy, max_lambda=20)
pi, policy_stable = policy_improvement(V, pi, A, probabilitiesx, probabilitiesy, 12,0.9)
while not policy_stable:
	V = policy_evaluation(V, pi, probabilitiesx, probabilitiesy, max_lambda=20)
	pi, policy_stable = policy_improvement(V, pi, A, probabilitiesx, probabilitiesy, 12,0.9)

sns.heatmap(pi)
plt.show()