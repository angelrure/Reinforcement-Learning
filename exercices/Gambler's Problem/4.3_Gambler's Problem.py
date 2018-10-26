import numpy as np
import matplotlib.pyplot as plt

def initialize(states = 100, max_reward = 1):
    """
    Creates the arbitrary set of initial states, last one is put to 
    the winning reward as it is the dummy state of already won
    """
    V = np.zeros(states+1)
    V[-1] = max_reward
    return V

def compute_Bellman(V, s, pheads, gamma):
    """
    Computes the Bellman iteration rule with the current set of states V, 
    for the specied state s. It uses the probabilities of winning as pheads and
    gamma as the discounting factor.
    """
    tempV = []
    for a in range(1,min(100-s, s)+1):
        tempV.append(pheads*(0 + gamma*V[s+a]) + (1-pheads)*(0+gamma*V[s-a]))
    return tempV

def value_iteration(V, pheads, gamma = 1, sigma = 0.1):
    """
    This function computes the value iteration algorithm for a set of 
    states V, using the winning probability of pheads, the discouting 
    factor gamma. Sigma is a threshold that ensures convergence as we keep 
    iterating until the maximum difference between all values and its 
    corresponding newly predicted is less than sigma. 
    """
    while True:
        delta = 0
        for s in range(1,len(V)-1):
            v = V[s]
            expected_value = compute_Bellman(V, s, pheads, gamma)
            V[s] = max(expected_value)
            delta = max(delta, abs(v - V[s]))
        print(delta)
        if delta > sigma:
            continue
        else:
            return V

def retrieve_policy(V, pheads, gamma):
    """
    This function takes the set of states V, the probability of winning 
    pheads and the discounting factor gamma and retrieves the optimal policy
    that the set of states describe by using a greedy approach. This works 
    because the action that maximies the next value at each step builds the 
    optimal policy if the state sets is optimal.
    """
    pi = np.zeros(100)
    for s in range(1,len(pi)):
        expected_value = compute_Bellman(V, s, pheads, gamma)
        pi[s] = np.argmax(expected_value) + 1
    return pi

V = initialize(max_reward = 10e10)
V = value_iteration(V, pheads = 0.25, sigma = 10e-20, gamma = 1)
pi = retrieve_policy(V, pheads = 0.25, gamma = 1)

plt.plot(V/max(V))
plt.xlabel('States (Current Money)')
plt.ylabel('Value (V(s))')
plt.title('State-value function')
plt.show()

plt.bar(range(100), height = pi)
plt.xlabel('States (Current Money)')
plt.ylabel('Actions (Bets)')
plt.title('Policy Pi')
plt.show()

V = initialize(max_reward = 10e10)
V = value_iteration(V, pheads = 0.55, sigma = 10e-20, gamma = 1)
pi = retrieve_policy(V, pheads = 0.55, gamma = 1)

plt.plot(V/max(V))
plt.xlabel('States (Current Money)')
plt.ylabel('Value (V(s))')
plt.title('State-value function')
plt.show()

plt.bar(range(100), height = pi)
plt.xlabel('States (Current Money)')
plt.ylabel('Actions (Bets)')
plt.title('Policy Pi')
plt.show()