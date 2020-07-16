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
        self.dt = 0.01 #[s]
        self.max_laser_length = 10

        # Starting position
        self.x_initial = -0. # [m]
        self.z_initial = 30. # [m]
        self.createDroneDictionary()

    def createDroneDictionary(self):
        # Create dictionary to hold all the relavent data for the drone parent class, to pass easily into child classes
        # x_initial, z_initial, gravity, mass, length, height, lasers, laser_range, input_limit, dt
        self.drone_dict = {"x_initial":self.x_initial,
        "z_initial":self.z_initial,
        "gravity":self.gravity,
        "mass":self.mass,
        "length":self.length,
        "height":self.height,
        "lasers":self.lasers,
        "laser_range":self.laser_range,
        "input_limit":self.input_limit,
        "dt":self.dt
         }
