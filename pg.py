import sys, gym
import numpy as np
from tqdm import *
from matplotlib import pyplot as plt

EPISODES = 5 if len(sys.argv) < 2 else int(sys.argv[1])

GAMMA = 1.0
ALPHA_CRITIC = 0.1
ALPHA_GP = 0.01

def softmax(st, th):
   result = np.exp(np.dot(th, st))
   sum_result = sum(result)
   return [ result[i] / sum_result for i in range(len(result)) ]

def policy(st, th):
   prob = softmax(st, th)
   return np.random.choice(len(prob), p=prob)

def value(st, w, a=None):
   result = np.dot(w, st)
   return result if a is None else result[a]

#env = gym.make('CartPole-v1')
env = gym.make('CartPole-v0')

th = np.ones(shape=(2, 4))
w = np.ones(shape=(2, 4))

total_reward = []

for i in tqdm(range(EPISODES)):
   state = env.reset()
   done = False
   cum_reward = 0

   while not done:

      action = policy(state, th)

      new_state, reward, done, _ = env.step(action)
      cum_reward += reward

      if done:
         td_target = reward
      else:
         td_target = reward + (GAMMA * value(new_state, w))
      td_error = td_target - value(state, w)

      #update critic
      w[action] += ALPHA_CRITIC * td_error * state

      #update policy
      prob = softmax(state, th)

      print('sum = ', sum([ prob[i] * np.dot(th[i], state) for i in range(len(prob)) ]))
      print('log gradient = ', value(state, action) - sum([ prob[i] * np.dot(th[i], state) for i in range(len(prob))]))
      """
      th[action] += ALPHA_GP * GAMMA * 
            (value(state, action) - sum([ prob[i] * np.dot(th[i], state) for i in range(len(prob))]))
      """
