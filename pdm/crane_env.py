#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:33:46 2021

@author: oscarjenot

Adapted from Open AI Gym : https://github.com/openai/gym/tree/master/gym/envs/classic_control

Reference:
Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016). Openai gym.
"""

"""
CRANE-v0 Environement
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class CraneEnv(gym.Env):

      
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 1  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        
        #self.goal_position = self.np_random.uniform(low=-2, high=2)
        self.goal_position = 1.0
        
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360 + math.pi # 12 degrees + pi rad
        #min theta_threshold_radian would be : self.theta_threshold_radians - 2 * 12 * 2 * math.pi / 360
        self.x_threshold = 2.4 
        
        # Angle limit set to 12 degrees + theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians + 12 * 2 * math.pi / 360,
                         np.finfo(np.float32).max],
                        dtype=np.float32)
        low = np.array([-self.x_threshold * 2,
                         -np.finfo(np.float32).max,
                         self.theta_threshold_radians - 2 * 12 * 2 * math.pi / 360,
                         -np.finfo(np.float32).max],
                        dtype=np.float32)
        
        self.action_space = spaces.Discrete(3)
        # [0, 1, 2] is action space
        
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        # [x, x_dot, theta; theta_dot] is observation space
        
        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        x, x_dot, theta, theta_dot = self.state
        force = (action - 1) * self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
       
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        
        self.state = (x, x_dot, theta, theta_dot)
        
        #REWARDS
        
        reward = -math.log(abs(self.goal_position - x)) - 1     
        
        if abs(self.goal_position - x) < 0.2 :
            reward = reward - math.log((abs(theta_dot) + abs(sintheta))) * 10 + 1
        
        if x > self.goal_position - 0.05 and x < self.goal_position + 0.05 and x_dot > - 0.05 and x_dot <   0.05 and theta_dot > - 0.03  and theta_dot <   0.03 and theta > math.pi  - 0.03 and theta <  math.pi + 0.03 : 
            reward = reward + 100000.0
        
        done = bool(  x_dot > -0.05 and x_dot < 0.05 and x > self.goal_position - 0.05 and x < self.goal_position + 0.05 and theta_dot > -0.03 and theta_dot < 0.03  and theta > math.pi - 0.03 and theta < math.pi + 0.03 )
                
        return np.array(self.state), reward, done, {}
    
    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-2, high=-1), self.np_random.uniform(low=-0.05, high=0.05), self.np_random.uniform(low=math.pi-0.05, high=math.pi+0.05), self.np_random.uniform(low=-0.05, high=0.05) ])
        #Try with initial conditions thet model was not trained on
        #self.state = np.array([self.np_random.uniform(low=2, high=4), self.np_random.uniform(low=-0.05, high=0.05), self.np_random.uniform(low=math.pi-0.05, high=math.pi+0.05), self.np_random.uniform(low=-0.05, high=0.05) ])
        return self.state
    
    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 300  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)
            
            flagx = (self.goal_position) * scale + screen_width / 2.0
            flagy1 = (carty - cartheight - polelen )
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)


            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        
            	
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            

      
