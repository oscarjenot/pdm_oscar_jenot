#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 18:34:11 2021

@author: oscarjenot

Adapted from Open AI Gym : https://github.com/openai/gym/tree/master/gym/envs/classic_control

Reference:
Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016). Openai gym.
"""


"""
CRANE-v2 Environement
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from numpy.linalg import inv

class CraneEnv2(gym.Env):


    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }


#_________________________________________________________________________________________________________________________________________________________________
    
    def __init__(self):
        
        #Constants
        self.gravity = 9.81
        self.m0      = 100.0         # mass cart
        self.m1      = 0.1           # mass rope 
        self.m2      = 5.0           # mass load
        self.L1      = 1.0           # lenght rope
        #L2 = state[6]               # lenght load (in observation space)
        
        #Integration for next states
        self.tau = 0.02                 # seconds between state updates
        self.kinematics_integrator = 'euler'
        #self.kinematics_integrator = 'semi-implicit euler'
        
        #Target
        self.x_goal_position = 1
        
        #thresholds
        self.x_threshold = 2.4  
        
        #Action space 
        """
        UNCOMMENT FOR CONTINUOUS ACTION SPACE
        
        self.min_action = -5 # min cart force, min rope force
        self.max_action = 5  # max cart force, max rope force
        
        self.action_space = spaces.Box(low = self.min_action, 
                                       high = self.max_action, 
                                       dtype=np.float.32)  # cart
        """

        #COMMENT FOR USE OF CONTINUOUS ACTION SPACE
        self.action_space = spaces.Discrete(11)
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] is action space


        
        # Observation space and limits
        
        self.x_min = -(self.x_threshold * 2)
        self.x_max = self.x_threshold * 2
        self.theta_min = -np.finfo(np.float32).max
        self.theta_max = np.finfo(np.float32).max
        self.velocity_min = -np.finfo(np.float32).max
        self.velocity_max = np.finfo(np.float32).max
        self.L2_min = 0
        self.L2_max = 1
    
        
        self.min_observation = np.array([self.x_min, self.velocity_min, 
                                               self.theta_min, self.velocity_min,
                                               self.theta_min, self.velocity_min,                       
                                               self.L2_min ], 
                                              dtype=np.float32)
        
        self.max_observation = np.array([self.x_max, self.velocity_max, 
                                               self.theta_max, self.velocity_max,
                                               self.theta_max, self.velocity_max,
                                               self.L2_max], 
                                              dtype=np.float32)
        
        self.observation_space = spaces.Box(low = self.min_observation,
                                            high = self.max_observation,
                                            dtype=np.float32)
        # [x, x_dot, theta1; theta1_dot, theta2, theta2_dot, L2] is observation space
        
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

#_________________________________________________________________________________________________________________________________________________________________
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
#_________________________________________________________________________________________________________________________________________________________________
    
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot, L2  = self.state
        
        
        """
        UNCOMMENT FOR CONTINUOUS ACTION SPACE
        
        force_cart = (action)   
        """
        force = action - 5
        # [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] is force action space
        
        # For the interested reader the dynamics of the system can be found here:
        # https://www.researchgate.net/publication/250107215_Optimal_Control_of_a_Double_Inverted_Pendulum_on_a_Cart
        
        # Constants for matrix form of Lagrange Equations
        m0 = self.m0
        m1 = self.m1
        m2 = self.m2
        L1 = self.L1
        L2 = L2
        
        d1 = m0 + m1 + m2
        d2 = (0.5 * m1 + m2 ) * L1
        d3 = 0.5 * m2 * L2
        d4 = (1/3 * m1 + m2) * L1**2
        d5 = 0.5 * m2 * L1 * L2
        d6 = 1/3 * m2 * L2**2
        f1 = (0.5 * m1 + m2) * L1 * self.gravity
        f2 = 0.5 * m2 * L2 * self.gravity
        
        costheta1 = math.cos(theta1)
        costheta2 = math.cos(theta2)
        costheta12 = math.cos(theta1 - theta2)  
        sintheta1 = math.sin(theta1)
        sintheta2 = math.sin(theta2)
        sintheta12 = math.sin(theta1 - theta2)
        
        #Matrix for Lagrange Equations : D(q) * q_acc + C(q, q_dot) * q_dot + G(q) = H * force
        
        D =      np.array([[d1           , d2*costheta1  , d3*costheta2   ], 
                           [d2*costheta1 , d4            , d5*costheta12  ],
                           [d3*costheta2 , d5*costheta12 , d6             ]])
        
        C =      np.array([[0 , -d2*sintheta1*theta1_dot , -d3*sintheta2*theta2_dot], 
                           [0 , 0                        , d5*sintheta12*theta2_dot],
                           [0 , -d5*sintheta12*theta1_dot, 0                       ]])
        
        G =      np.array([0 , -f1*sintheta1 , -f2*sintheta2 ])  
        
        H =      np.array([1, 0, 0]) 
        
        D_inv = inv(D) #invers of matrix D
        
        #ODEs q_acc = (x_acc, theta1_acc, theta2_acc)
        # q_acc = D^(-1) * (-C*q_dot - G - H*force) 
        
        
        q_dot =   np.array([x_dot,
                            theta1_dot,
                            theta2_dot])
                
        q_acc = np.dot( D_inv , ( -np.dot( C , q_dot ) -G - H*force ))
        
        x_acc      = q_acc[0]
        theta1_acc = q_acc[1]
        theta2_acc = q_acc[2]
        
        # Finding next state
     
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * x_acc
            theta1 = theta1 + self.tau * theta1_dot
            theta1_dot = theta1_dot + self.tau * theta1_acc
            theta2 = theta2 + self.tau * theta2_dot
            theta2_dot = theta2_dot + self.tau * theta2_acc

            
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * x_acc
            x = x + self.tau * x_dot
            theta1_dot = theta1_dot + self.tau * theta1_acc
            theta1 = theta1 + self.tau * theta1_dot
            theta2_dot = theta2_dot + self.tau * theta2_acc
            theta2 = theta2 + self.tau * theta2_dot
            
        
        #REWARDS
        sintheta1 = math.sin(theta1)
        sintheta2 = math.sin(theta2)
        
        flag = self.x_goal_position
        
        reward = -1
        
        reward = reward -np.log(abs(flag - x))*3 # x position reward
        
        reward = reward - abs(theta1 % math.pi)
        reward = reward - abs(theta2 % math.pi)
        
        
        if abs(flag - x) < 0.2 :
            reward = reward - np.log((abs(theta1_dot) + abs(sintheta1))) * 10 + 1
            reward = reward - np.log((abs(theta2_dot) + abs(sintheta2))) * 10 + 1
        
            
        
        if (        x     > flag - 0.1  and x          < flag + 0.1
                and x_dot      > - 0.1  and x_dot      <   0.1
                and theta1_dot > - 0.1  and theta1_dot <   0.1
                and sintheta1  > - 0.1  and sintheta1  <   0.1 
                and theta2_dot > - 0.1  and theta2_dot <   0.1
                and sintheta2  > - 0.1  and sintheta2  <   0.1 
                ):
            reward = reward + 100000.0
            
        done = bool(x     > flag - 0.1  and x          < flag + 0.1
                and x_dot      > - 0.1  and x_dot      <   0.1
                and theta1_dot > - 0.1  and theta1_dot <   0.1
                and sintheta1  > - 0.1  and sintheta1  <   0.1 
                and theta2_dot > - 0.1  and theta2_dot <   0.1
                and sintheta2  > - 0.1  and sintheta2  <   0.1 
                )
        
        #Limits on observation values
        
        if theta1_dot > 10**20 :
            print('!!! Theta dimension problem !!!')
            theta1 = self.np_random.uniform(low=-math.pi, high=math.pi)
            theta1_dot = self.np_random.uniform(low=-0.1, high=0.1)
            theta1_acc = 0
            #x_dot = 0
            #xacc = 0
            #reward = reward -10000
            done = True
            
        if theta2_dot > 10**20 :
            print('!!! Theta dimension problem !!!')
            theta2 = self.np_random.uniform(low=-math.pi, high=math.pi)
            theta2_dot = self.np_random.uniform(low=-0.1, high=0.1)
            theta2_acc = 0
            #x_dot = 0
            #xacc = 0
            #reward = reward -10000
            done = True
          
        self.state = (x, x_dot, theta1, theta1_dot, theta2, theta2_dot, L2)

        
        return np.array(self.state, dtype='float32'), reward, done, {}
   
    
   
    
#_________________________________________________________________________________________________________________________________________________________________
   
    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-2, high=-1)                    , self.np_random.uniform(low=-0.05, high=0.05), # x and x_dot
                               self.np_random.uniform(low=math.pi-0.05, high=math.pi+0.05), self.np_random.uniform(low=-0.05, high=0.05), # theta1 and thetha1_dot
                               self.np_random.uniform(low=math.pi-0.05, high=math.pi+0.05), self.np_random.uniform(low=-0.05, high=0.05), # theta2 and thetha2_dot
                               1.0])              # L2 lenght of the load
                        
        self.steps_beyond_done = None
        return self.state
    
    
#_________________________________________________________________________________________________________________________________________________________________
        
    def render(self, mode='human'):
      
        if self.state is None:
            return None

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        
#_________________________________________________________________________________________________________________________________________________________________
            	
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            

      
