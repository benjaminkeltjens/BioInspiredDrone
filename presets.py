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
        self.drag_coeff = 0.5
        self.mass = 5 # [kg]
        self.length = 0.3 # [m]
        self.height = 0.05 # [m]
        self.lasers = 8
        self.laser_range = 2*np.pi # [rad]
        self.input_limit = 50 # [N]
        self.input_rate_limit = 500000 # [N/s]
        self.dt = 0.01 #[s]
        self.max_laser_length = 10
        self.safe_vel = 1.0 # [m/s] # Safe velocity to touchdown
        self.safe_angle = 25*np.pi/180 # [rad] # Safe angle from 0 to touchdown with

        # Starting position
        self.x_initial = -0. # [m]
        self.z_initial = 30. # [m]
        self.theta_intial = 0*np.pi/180 # [rad]
        self.createDroneDictionary()

    def createDroneDictionary(self):
        # Create dictionary to hold all the relavent data for the drone parent class, to pass easily into child classes
        # x_initial, z_initial, gravity, mass, length, height, lasers, laser_range, input_limit, dt
        self.drone_dict = {"x_initial":self.x_initial,
        "z_initial":self.z_initial,
        "theta_intial":self.theta_intial,
        "gravity":self.gravity,
        "drag_coeff":self.drag_coeff,
        "mass":self.mass,
        "length":self.length,
        "height":self.height,
        "lasers":self.lasers,
        "laser_range":self.laser_range,
        "input_limit":self.input_limit,
        "input_rate_limit":self.input_rate_limit,
        "dt":self.dt
         }

    def writeCTRNNConfigFile(self):
        fin = open("config-ctrnn-template")
        fout = open("config-ctrnn", "wt")
        for line in fin:
            fout.write(line.replace("num_inputs              = 8", "num_inputs              = "+str(4)))
        fin.close()
        fout.close()

    def writeNNConfigFile(self):
        fin = open("config-nn-template")
        fout = open("config-nn", "wt")
        for line in fin:
            fout.write(line.replace("num_inputs              = 8", "num_inputs              = "+str(4)))
        fin.close()
        fout.close()
