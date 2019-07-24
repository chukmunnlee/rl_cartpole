import numpy as np
import gym, sys
from tqdm import *
from matplotlib import pyplot as plt

np.random.seed(42)

EPISODES = 5 if len(sys.argv) < 2 else int(sys.argv[1])

GAMMA = 1.0
ALPHA = 0.1
EPSILON = 0.01

def plot_it(tot_steps, ave_reward, lines, pause_time=0.1):

   if lines == []:
      plt.ion()
      fig = plt.figure(figsize=(10, 10))
      avgRewAx = fig.add_subplot(211)
      avgRewAx.set_xlabel('Episodes')
      avgRewAx.set_ylabel('Avg Reward/episode')
      avgRewLine, = avgRewAx.plot(range(len(ave_reward)), ave_reward, 'go-')

      stepsAx = fig.add_subplot(212)
      stepsAx.set_xlabel('Episodes')
      stepsAx.set_ylabel('Steps')
      stepsLine, = stepsAx.plot(range(len(total_steps)), total_steps, 'bx-')

      plt.show()

   else:
      stepsLine, avgRewLine, stepsAx, avgRewAx = lines

   avgRewLine.set_data(range(len(ave_reward)), ave_reward)
   avgRewAx.set_xlim([0,  len(ave_reward)])
   if (np.max(ave_reward) >= avgRewLine.axes.get_ylim()[1]):
      avgRewAx.set_ylim([0,  np.max(ave_reward) + np.std(ave_reward)])

   stepsLine.set_data(range(len(total_steps)), total_steps)
   stepsAx.set_xlim([0, len(total_steps)])
   if (np.max(total_steps) > stepsLine.axes.get_ylim()[1]):
      stepsAx.set_ylim([0, np.max(total_steps) + np.std(total_steps)])

   plt.pause(pause_time)

   return [stepsLine, avgRewLine, stepsAx, avgRewAx]

plt.style.use('ggplot')

env = gym.make('CartPole-v0')
env = gym.make('CartPole-v1')

def Q(st, w, a=None):
   predict = np.dot(w, st) 
   return predict if a is None else predict[a]

def policy(st, w):
   if (np.random.rand() > EPSILON):
      return np.argmax(Q(st, w))
   return env.action_space.sample()

#w = np.random.rand(2, 4)
w = np.random.rand(2, 4)
#w = np.array([[0.94523601, 0.99865513, 0.15720279, 0.4700091 ], [0.06710925, 0.40331484, 0.81231174, 0.38467417]]) * 100
w = np.ones(shape=(2, 4))

lines = []
total_reward = []
ave_reward = []
total_steps = []

#for i in tqdm(range(EPISODES)):
for i in range(EPISODES):
   state = env.reset()
   action = policy(state, w)
   done = False
   cum_reward = 0
   steps = 0

   while not done:

      env.render()

      new_state, reward, done, _ = env.step(action)
      steps += 1
      cum_reward += reward

      if done:
         print('Episode: %d, reward: %d' %(i, cum_reward), end='\r', flush=False)
         td_target = reward
         total_reward.append(cum_reward)
         ave_reward.append(sum(total_reward) / (i + 1))
         total_steps.append(steps)
      else:
         new_action = policy(new_state, w)
         td_target = reward - (GAMMA * Q(new_state, w, new_action))

      td_error = td_target - Q(state, w, action)
      w[action] += (ALPHA * td_error * state)

      state = new_state
      action = new_action

   lines = plot_it(total_steps, ave_reward, lines)

env.close()

print('Average reward: %.3f' %(sum(total_reward) / len(total_reward)))

plt.plot(range(len(total_steps)), total_reward, label='Steps/ep')
plt.plot(range(len(ave_reward)), ave_reward, label='Average reward / ep')

plt.title('Average reward: %.3f, Episodes: %d' %(sum(total_reward) / len(total_reward), EPISODES))

plt.legend()

plt.show()

