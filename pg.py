import sys, gym
import numpy as np
from tqdm import *
from matplotlib import pyplot as plt

EPISODES = 5 if len(sys.argv) < 2 else int(sys.argv[1])

GAMMA = 1
ALPHA_CRITIC = 0.2
#ALPHA_ACTOR = 0.0025
#ALPHA_ACTOR = 0.53
ALPHA_ACTOR = 0.6

def softmax(st, th):
   exp = np.exp(np.dot(th, st))
   return exp / np.sum(exp)

def policy(st, th):
   prob = softmax(st, th)
   return np.random.choice(len(prob), p=prob)

def value(st, w):
   return np.dot(w, st)

def gradient_check():
   pass

#env = gym.make('CartPole-v1')
env = gym.make('CartPole-v0')

np.random.seed(1)

#th = np.random.rand(2, 4) * 1000
th = np.random.rand(2, 4) 
w = np.ones(shape=(1, 4))

print('th = ', th)

total_reward = []
ave_reward = []

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
         print('Episode: %d , reward: %d ' %(i, cum_reward), end='\r', flush=False)
         td_target = reward
         total_reward.append(cum_reward)
         ave_reward.append(sum(total_reward) / (i + 1))

      else:
         td_target = reward + (GAMMA * value(new_state, w))

      td_error = td_target - value(state, w)

      # calculate softmax gradient
      prob = softmax(state, th)
      grad = np.dot(th[action], state) - sum([prob[p] * np.dot(th[p], state) for p in range(len(prob))])

      #update critic
      w += ALPHA_CRITIC * td_error * state

      #update policy
      th[action] += ALPHA_ACTOR * td_error * grad

      state = new_state

env.close()

print('Average reward: %.3f    ' %(sum(total_reward) / len(total_reward)))

plt.plot(range(len(total_reward)), total_reward, label='EPISODES %d' %EPISODES)
plt.plot(range(len(ave_reward)), ave_reward, label='Average reward / episode')
plt.title('Average reward: %.3f, Episodes: %d' %(sum(total_reward) / len(total_reward), EPISODES))
plt.legend()

plt.show()
