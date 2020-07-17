"""
This file is to handle preset data combinations for simulations

Benjamin Keltjens
July 2020
"""

import numpy as np

class Presets(object):

    def __init__(self):
        pass

    def loadDefault(self):
        self.gravity = -9.80665 # [m/s^2]
        self.mass = 5 # [kg]
        self.length = 0.3 # [m]
        self.height = 0.05 # [m]
        self.lasers = 8
        self.laser_range = 2*np.pi # [rad]
        self.input_limit = 50 # [N]
        self.input_rate_limit = 500 # [N/s]
        self.dt = 0.01 #[s]
        self.max_laser_length = 10
        self.safe_vel = 1 # [m/s] # Safe velocity to touchdown
        self.safe_angle = 15*np.pi/180 # [rad] # Safe angle from 0 to touchdown with

        # Starting position
        self.x_initial = -0. # [m]
        self.z_initial = 20. # [m]
        self.theta_intial = np.pi*0/180 # [rad]
        self.createDroneDictionary()

    def createDroneDictionary(self):
        # Create dictionary to hold all the relavent data for the drone parent class, to pass easily into child classes
        # x_initial, z_initial, gravity, mass, length, height, lasers, laser_range, input_limit, dt
        self.drone_dict = {"x_initial":self.x_initial,
        "z_initial":self.z_initial,
        "theta_intial":self.theta_intial,
        "gravity":self.gravity,
        "mass":self.mass,
        "length":self.length,
        "height":self.height,
        "lasers":self.lasers,
        "laser_range":self.laser_range,
        "input_limit":self.input_limit,
        "input_rate_limit":self.input_rate_limit,
        "dt":self.dt
         }
