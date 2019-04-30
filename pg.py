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

def value(st, w):
   return np.dot(w, st)

#env = gym.make('CartPole-v1')
env = gym.make('CartPole-v0')

#th = np.ones(shape=(2, 4))
th = np.random.rand(2, 4)
w = np.ones(shape=(1, 4))

print('th = ', th)

total_reward = []

#for i in tqdm(range(EPISODES)):
for i in range(EPISODES):
   state = env.reset()
   done = False
   cum_reward = 0

   while not done:

      action = policy(state, th)

      new_state, reward, done, _ = env.step(action)
      cum_reward += reward

      if done:
         #print('Episode: %d, reward: %d' %(i, cum_reward))
         td_target = reward
         total_reward.append(cum_reward)

      else:
         td_target = reward + (GAMMA * value(new_state, w))

      td_error = td_target - value(state, w)

      #update critic
      w += ALPHA_CRITIC * td_error * state

      #update policy
      prob = softmax(state, th)
#      print('prob = ', prob)
#      print('\tth = ', th)
#      print('\taction = ', action)
#      print('\tth[action] = ', th[action])
#      print('\tnp.dot = ', np.dot(th[action], state))
#      print('\texpectation = ', sum([ prob[p] * np.dot(th[p], state) for p in range(len(prob))] ))

      th[action] += ALPHA_GP * GAMMA * (
            np.dot(th[action], state) - sum([ prob[p] * np.dot(th[p], state) for p in range(len(prob))] ))

      state = new_state

env.close()

print('Average reward: %.3f' %(sum(total_reward) / len(total_reward)))

plt.plot(range(len(total_reward)), total_reward, label='EPISODES %d' %EPISODES)
plt.legend()

plt.show()
