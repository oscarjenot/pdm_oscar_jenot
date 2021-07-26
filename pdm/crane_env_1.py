#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mai 10  8 13:32:43 2021

@author: oscarjenot
Adapted from Open AI Gym : https://github.com/openai/gym/tree/master/gym/envs/classic_control

Reference:
Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016). Openai gym.
"""

"""
CRANE-v1 Environement
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class CraneEnv1(gym.Env):

    """
    Description
        An mass is attached to an inextensible rope attached by an un-actuated 
        joint to a cart, which moves along a frictionless track. 
        The pendulum starts downward, and the goal is to stabilysing 
        it at an other x-location of the cart (flag).
        
    Observation
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -2pi rad (-360 deg)    2pi rad (360 deg)
        3       Pole Angular Velocity     -Inf                    Inf
        4       Rope Lenght               0                       Inf
        5       Rope Velocity             -Inf                    Inf
        6       x goal position           -2                      2
        7       y goal position           0                       2  
        
    Coordinates : Positive angle is trigonometric angle, positive x position is 
    to the right, pole lenght is positive from the cart to the mass. Pole lowering 
    force is positive from mass to cart.        
        
    Actions:
        Type: Box(2)
        Num   Action
        0     Driving force. Type: Discrete(3): accelerate to the Left (0),
              no acceleration (1), accelerate to the right (2).
        1     Pole lowering and hoisting force. Type: Discrete(3): hoist Pple (0),
              no force (1), lower pole (2).
        
    Reward:
        Reward of 0 is awarded if the agent reached the flag with a small tolerance.
        Tolerance on pole lenght : [y_flag_dist -0.05 , y_flag_dist +0.05] Rad.
        Tolerance on other Observations : [-0.05 , 0.05].
        Reward of -1 is awarded if the position of the agent is not on the flag.      
    
    Starting State:
        Cart position is assigned a uniform random value in [-2 , 0].
        Cart velocity, pole angle and angular velocity are assigned a uniform random 
        value in [-0.05 , 0.05]. 
        Pole lenght is assigned a uniform random value in [0 , 2].        
        
    Episode Termination:
        Pole Angle is not longer in [-12, +12] degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        The agent reached the flag with a small tolerance (position to be set up).
        
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }


#_________________________________________________________________________________________________________________________________________________________________
    
    def __init__(self):
        
        #Constants
        self.gravity = 9.81
        self.masscart = 1.0
        self.masspoint = 0.1
        
        self.tau = 0.02  # seconds between state updates
        self.force_cart = 5.0
        self.force_rope = 1.0
        
        self.kinematics_integrator = 'euler'
        #self.kinematics_integrator = 'semi-implicit euler'
        
        
        self.cart_height = 2.5
        
        #thresholds
        self.theta_threshold_radians = 12 * 2 * math.pi / 360 #12 degrees
        self.x_threshold = 2.4
        
        
        #Action space 
        """
        UNCOMMENT FOR CONTINUOUS ACTION SPACE
        
        self.min_action = np.array([-1.0, -1.0]) # min cart force, min rope force
        self.max_action = np.array([1.0, 1.0])   # max cart force, max rope force
        
        self.action_space = spaces.Box(low = self.min_action, 
                                       high = self.max_action, 
                                       dtype=np.float.32)  # cart, rope
        """

        #COMMENT FOR USE OF CONTINUOUS ACTION SPACE
        self.action_space = spaces.Discrete(9)


        
        # Observation space and limits
        
        self.x_min = -(self.x_threshold * 2)
        self.x_max = self.x_threshold * 2
        self.theta_min = -(2*self.theta_threshold_radians)
        self.theta_max = 2*self.theta_threshold_radians
        self.l_min = 0
        self.l_max = np.finfo(np.float32).max
        self.velocity_min = -np.finfo(np.float32).max
        self.velocity_max = np.finfo(np.float32).max
        self.x_goal_min = -2
        self.x_goal_max =  2
        self.y_goal_min =  0
        self.y_goal_max =  2
        
        self.min_observation = np.array([self.x_min, self.velocity_min, 
                                               self.theta_min, self.velocity_min,
                                               self.l_min, self.velocity_min, 
                                               self.x_goal_min, self.y_goal_min], 
                                              dtype=np.float32)
        
        self.max_observation = np.array([self.x_max, self.velocity_max, 
                                               self.theta_max, self.velocity_max,
                                               self.l_max, self.velocity_max,
                                               self.x_goal_max, self.y_goal_max], 
                                              dtype=np.float32)
        
        self.observation_space = spaces.Box(low = self.min_observation,
                                            high = self.max_observation,
                                            dtype=np.float32)
        
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
        
        x, x_dot, theta, theta_dot, l, l_dot, x_goal, y_goal = self.state
        
        
        """
        UNCOMMENT FOR CONTINUOUS ACTION SPACE
        
        force_cart = (action[0]) * self.force_cart
        force_rope = (action[1]) * self.force_rope - self.masspoint * self.gravity #removing masspoint * gravity brings the mass to an equilibrium when thetha acc is zero
        """
        
        """
        FOR DISCRETE ACTION SPACE, THIS IS THE FORCE / ACTION MATRIX :
            
            [  ][-1][ 0][ 1] force_cart
            [-1][ 0][ 1][ 2] 
            [ 0][ 3][ 4][ 5] 
            [ 1][ 6][ 7][ 8]
           force_
           rope 
            
        """
        
        #Discrete action space / force mapping
        
        temp = np.array([[0, 1, 2], 
                         [3, 4, 5],
                         [6, 7, 8]])

        if action in temp[:,0] :
            force_cart = -1 * self.force_cart 
        if action in temp[:,1] :
            force_cart =  0 * self.force_cart
        if action in temp[:,2] :
            force_cart =  1 * self.force_cart
        
        if action in temp[0,:] :
            force_rope = -1 * self.force_rope - self.masspoint * self.gravity #removing masspoint * gravity brings the mass to an equilibrium when thetha acc is zero
        if action in temp[1,:] :
            force_rope =  0 * self.force_rope - self.masspoint * self.gravity #removing masspoint * gravity brings the mass to an equilibrium when thetha acc is zero
        if action in temp[2,:] :
            force_rope =  1 * self.force_rope - self.masspoint * self.gravity #removing masspoint * gravity brings the mass to an equilibrium when thetha acc is zero
                
               
        # For the interested reader the dynamics of the system can be found here:
        # https://www.researchgate.net/publication/261295749_Tracking_Control_for_an_Underactuated_Two-Dimensional_Overhead_Crane
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        temp = ( force_cart - force_rope * sintheta )
        
        
        xacc = temp / self.masscart
        thetaacc = -( temp * costheta / self.masscart + 2 * l_dot * theta_dot + self.gravity * sintheta ) / l                     
        lacc =  (force_rope / self.masspoint + (theta_dot)**(2) * l + self.gravity * costheta  - temp * sintheta / self.masscart)
        
     
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            l = l + self.tau * l_dot
            l_dot = l_dot + self.tau * lacc
            
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
            l_dot = l_dot + self.tau * lacc
            l = l + self.tau * l_dot
            
        
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        self.x_goal_position = x_goal
        self.y_goal_position = y_goal
        
        
        # REWARDS
        reward = -1
        
        reward = reward -np.log(abs(1.0 - x)) * 2
        
        
        if abs(1 - x) < 0.5 :
            
            reward = reward + 3
            if l > 3 :
                reward = reward  -l_dot / 6 
            if l < 2 : 
                reward = reward + l_dot / 6
            
            
            reward = reward - np.log((abs(theta_dot) + 3*abs(sintheta))) * 15 + 5

            reward = reward - np.log(abs(l - abs(self.cart_height))) * 8 + 5

             
            
            #if ((abs(theta_dot) + abs(sintheta))) < 0.1:
                #reward = reward - math.log(abs(l - abs(self.cart_height))) * 7 + 50
                #reward = reward - (abs(l - abs(self.cart_height))) + 30
            
            #reward = reward - math.log(abs(l - abs(self.cart_height))) * 10   +10           
            #reward = reward - abs(l - abs(self.cart_height))           
                                       
                                       
        '''
        #PRECISE GOAL
        
        if (x > self.x_goal_position - 0.05     and x < self.x_goal_position + 0.05
            and x_dot > - 0.05         and x_dot <   0.05
            and theta_dot > - 0.05     and theta_dot <   0.05
            and sintheta > - 0.05         and sintheta <   0.05
            and l_dot > - 0.05         and l_dot <   0.05
            and l > (self.cart_height - self.y_goal_position) - 0.05  and l < (self.cart_height - self.y_goal_position) + 0.05
            ): 
            reward = reward + 200000.0
        
        
        done = bool(
            x > self.x_goal_position - 0.05     and x < self.x_goal_position + 0.05
            and x_dot > - 0.05         and x_dot <   0.05
            and theta_dot > - 0.05     and theta_dot <   0.05
            and sintheta > - 0.05         and sintheta <   0.05
            and l_dot > - 0.05         and l_dot <   0.05
            and l > (self.cart_height - self.y_goal_position) - 0.05
            and l < (self.cart_height - self.y_goal_position) + 0.05
        )
        '''       
        
        if (x > self.x_goal_position - 0.15     and x < self.x_goal_position + 0.15
            and x_dot > - 0.15         and x_dot <   0.15
            and theta_dot > - 0.15     and theta_dot <   0.15
            and sintheta > - 0.15         and sintheta <   0.15
            and l_dot > - 0.15         and l_dot <   0.15
            and l > 2.5 - 0.15  and l < 2.5 + 0.15
            ): 
            reward = reward + 10000000.0
            
        if (x > self.x_goal_position - 0.2     and x < self.x_goal_position + 0.2
            and x_dot > - 0.2         and x_dot <   0.2
            and theta_dot > - 0.2     and theta_dot <   0.2
            and sintheta > - 0.2         and sintheta <   0.2
            #and l_dot > - 0.1         and l_dot <   0.1
            and l > 2.5 - 0.3  and l < 2.5 + 0.3
            ): 
            reward = reward + 10000.0    
            #print('First l_position Reward !')
        
        
        done = bool(
            x > self.x_goal_position - 0.15     and x < self.x_goal_position + 0.15
            and x_dot > - 0.15         and x_dot <   0.15
            and theta_dot > - 0.15     and theta_dot <   0.15
            and sintheta > - 0.15      and sintheta <   0.15
            and l_dot > - 0.15         and l_dot <   0.15
            and l > 2.5 - 0.15         and l < 2.5+ 0.15
        )        
        
        
        #Limits on observation values
        
        if theta_dot > 10**20 :
            #print('!!! Theta dimension problem !!!')
            #print(self.state)
            theta = self.np_random.uniform(low=-math.pi, high=math.pi)
            theta_dot = self.np_random.uniform(low=-0.1, high=0.1)
            thetaacc = 0
            x_dot = 0
            xacc = 0
            #reward = reward - 2000
            #CraneEnv1.reset(self)
            #x, x_dot, theta, theta_dot, l, l_dot, x_goal, y_goal = self.state
            #done = True
            
        if  l_dot  > 10**20 :
            #print('!!! Lenght dimension problem !!!')
            #print(self.state)
            #reward = reward - 2000
            theta = 1.0
            theta_dot = self.np_random.uniform(low=-math.pi, high=math.pi)
            thetaacc = 0
            x_dot = 0
            xacc = 0
            l = 1.0
            l_dot = 0
            lacc = 0
            #CraneEnv1.reset(self)
            #x, x_dot, theta, theta_dot, l, l_dot, x_goal, y_goal = self.state
            #done = True
           
        if l < 0.1 :
            #print('!!! Lenght Can not be negatif !!!')
            #reward = reward - 10
            #print(self.state)
            l = 0.1 
            l_dot = 0
            lacc = 0
            theta = self.np_random.uniform(low=-0.2, high=0.2)
            theta_dot = 0
            thetaacc = 0
            #done = True
        
        '''
        if l < 0.1 :
            #print('!!! Lenght Can not be negatif !!!')
            #reward = reward - 10
            #print(self.state)
            l = 0.2 
            l_dot = 0
            lacc = 0
            theta = self.np_random.uniform(low=-0.2, high=0.2)
            theta_dot = 0
            thetaacc = 0
            done = True
        '''
        
        #if x_dot > 10**20 :
            #print('!!! x_position dimension problem !!!')
            #print(self.state)
            #reward = reward - 2000
            #CraneEnv1.reset(self)
            #x, x_dot, theta, theta_dot, l, l_dot, x_goal, y_goal = self.state
            #done = True
            
            
        self.state = (x, x_dot, theta, theta_dot, l, l_dot, x_goal, y_goal)

        
        return np.array(self.state), reward, done, {}
   
    
   
    
#_________________________________________________________________________________________________________________________________________________________________
   
    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-2, high=-1), self.np_random.uniform(low=-0.05, high=0.05), # x and x_dot
                               self.np_random.uniform(low=-0.05, high=0.05), self.np_random.uniform(low=-0.05, high=0.05), # theta and thetha_dot
                               self.np_random.uniform(low=1, high=2), self.np_random.uniform(low=-0.05, high=0.05), # l and l_dot
                               self.np_random.uniform(low=1.0, high=1.0), self.np_random.uniform(low=0.0, high=0.0)])  #x_goal and y_goal
                        
        self.x_goal_position = self.state[6]
        self.y_goal_position = self.state[7]

        self.steps_beyond_done = None
        return self.state
    
    
#_________________________________________________________________________________________________________________________________________________________________
        
    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        
        carty = self.cart_height * scale  + screen_height/20 # screen_height/20 is the vertical offset
        polewidth = 2.0
        polelen = scale * (self.state[4])
        cartwidth = 50.0
        cartheight = 30.0
        
        
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            #pole.set_color(.8, .6, .4)
            pole.set_color(.5, .5, .5)
            self.poletrans = rendering.Transform()
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            
            self.axle = rendering.make_circle(polewidth * 2)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            
            self.mass = rendering.make_circle(self.masspoint * 100)
            self.mass.set_color(0, 0, 0)
            self.masstrans = rendering.Transform()
            self.mass.add_attr(self.masstrans)
            self.viewer.add_geom(self.mass)
            
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0.8, 0, 0)
            self.viewer.add_geom(self.track)
            
            flagx = (self.x_goal_position) * scale + screen_width / 2.0
            flagy1 = (scale * self.y_goal_position + screen_height/20 ) # screen_height/20 is the vertical offset
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

            self.ground = rendering.Line((0, 0 + screen_height/20), (screen_width, 0 + screen_height/20))
            self.ground.set_color(0, 0, 0)
            self.viewer.add_geom(self.ground)

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
        self.poletrans.set_rotation(x[2]+ math.pi)
        
        massx = cartx + (math.sin(x[2]) * x[4])*scale
        massy = carty - (math.cos(x[2]) * x[4])*scale
        self.masstrans.set_translation(massx, massy)
        

       

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        
#_________________________________________________________________________________________________________________________________________________________________
            	
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            

      
