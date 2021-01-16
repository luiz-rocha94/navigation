# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 17:37:00 2021

@author: rocha
"""

from unityagents import UnityEnvironment
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch

env = UnityEnvironment(file_name=r"D:\deep-reinforcement-learning\p1_navigation\Banana_Windows_x86_64\Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain      = env.brains[brain_name]

#Create agent brain
from navigation_agent import Agent
agent = Agent(state_size=37, action_size=4, seed=0)

#setup
n_episodes    = 1000                     # number of training episodes
scores        = []                       # list containing scores from each episode
scores_window = deque(maxlen=100)        # last 100 scores
eps           = 1                        # starting value of epsilon
eps_end       = 0.01                     # minimum value of epsilon
eps_decay     = eps_end**(1/n_episodes)  # decreasing epsilon

for i_episode in range(n_episodes):
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    score    = 0                                       # initialize the score
    while True:
        state      = env_info.vector_observations[0] # get the current state
        action     = agent.act(state, eps)           # select an action
        env_info   = env.step(action)[brain_name]    # send the action to the environment
        next_state = env_info.vector_observations[0] # get the next state
        reward     = env_info.rewards[0]             # get the reward
        done       = env_info.local_done[0]          # see if episode has finished
        agent.step(state, action, reward, 
                   next_state, done)                 # Save experience and learn 
        score     += reward                          # update the score
        state      = next_state                      # roll over the state to next time step
        if done:                                     # exit loop if episode finished, done with 300 time stamps
            break
    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score
    eps = max(eps_end, eps_decay*eps) # decrease epsilon
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    if np.mean(scores_window)>=13.0:
        print('\nEnvironment solved in episode {:d}!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        torch.save(agent.qnetwork_local.state_dict(), 'model.pth')
        break    


#View Agent
env_info = env.reset(train_mode=False)[brain_name] # reset the environment
score    = 0                                       # initialize the score
while True:
    state      = env_info.vector_observations[0]   # get the current state
    action     = agent.act(state, eps)             # select an action
    env_info   = env.step(action)[brain_name]      # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward     = env_info.rewards[0]               # get the reward
    done       = env_info.local_done[0]            # see if episode has finished
    agent.step(state, action, reward, 
               next_state, done)                   # Save experience and learn 
    score     += reward                            # update the score
    state      = next_state                        # roll over the state to next time step
    print('\rScore {}'.format(score), end="")
    if done:                                       # exit loop if episode finished, done with 300 time stamps
        break


env.close()    

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()