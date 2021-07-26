#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 23:19:10 2021
	
@author: oscarjenot
"""

import gym
import time

env = gym.make('crane-v0')

observation = env.reset()
reward_sum = 0
for t in range(200):
    env.render()
    time.sleep(0.01)
    action = env.action_space.sample()
    #action = 1			
    observation, reward, done, info = env.step(action)
    #print(action)
    #print(observation)
    #print('reward at step', t, 'is:', reward)
    reward_sum = reward_sum + reward
    
    #print ('reward sum is', reward_sum)
    
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()




