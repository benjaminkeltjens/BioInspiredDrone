"""
This file describes the parent drone class, and child drone classes with different control systems

Benjamin Keltjens
July 2020
"""
import numpy as np

class Drone(object):
    # The parent drone describes the kinematics and dynamics of the drone, handles inputs

    def __init__(self, xposition, zposition, gravity, mass, length, height, lasers, laser_range, dt):
        # Describe the drone charactertics

        self.mass = mass
        self.length = length
        self.height = height
        self.inertia = None # TODO: calculate inertia
        self.lasers = lasers # number of lasers
        self.laser_range = laser_range
        self.laser_list = [None] * lasers
        self.laser_distance = [None] * lasers
        self.dt = dt

        # Initialise the drone static at a given location

        self.pos = np.array([[xposition], [zposition]]) # x,z
        self.theta_pos = 0
        self.vel = np.array([[0], [0]])
        self.theta_vel = 0
        self.accel = np.array([[0], [-gravity]])
        self.theta_accel = 0
        self.shape = None
        self.updateShape()

        self.gravity = gravity


    def update(self, input_L, input_R):
        # Find dynamics, kinematics, and update states

        # TODO: implement limits on rate change of input forces

        forces, moment = self.resolveDynamics(input_L, input_R)
        self.accel = forces/self.mass
        self.theta_accel = moment/self.inertia

        # Euler Integration
        self.vel += self.accel * self.dt
        self.theta_vel += self.theta_accel * self.dt

        self.pos += self.vel * self.dt
        self.theta_pos += self.theta_vel * self.dt
        self.theta_pos = wrapAngle(self.theta_pos, 2*np.pi) # wrap angle to stay between bounds of 0 and 2pi

        self.updateLaserAngles()


    def resolveDyanmics(self, input_L, input_R):
        # Find linear and rotational dynamic from input
        body_thrust = input_L + input_R
        global_thrust = np.array([[body_thrust*np.sin(self.theta_pos)], [body_thrust*np.cos(self.theta_pos)]])
        global_forces = global_thrust + np.array([[0], [-self.gravity*self.mass]])

        moment = self.length * (input_L - input_R) * 0.5

        return global_forces, moment

    def updateLaserAngles(self):
        # Update list of the laser angles ordered in the negative theta direction

        for i in range(self.lasers):
            laser_angle = self.theta_pos + (np.pi/2) + (self.laser_range/(self.lasers-1))*(((lasers-1)/2)-i)
            self.laser_list[i] = wrapAngle(laser_angle)

    def wrapAngle(self, angle, maximum):
        # Wrap angle between 0 and maximum
        return (angle%maximum)

    def updateShape(self):
        # Update drone shape representation

        # Create vectors to go from centre location to four corners
        v1 = np.array([[-(self.length/2)*np.cos(self.theta_pos)],[(self.length/2)*np.sin(theta_pos)]])
        v2 = np.array([[(self.height/2)*np.sin(self.theta_pos)],[(self.height/2)*np.cos(theta_pos)]])

        P1 = self.pos+v1+v2
        P2 = self.pos-v1+v2
        P3 = self.pos-v1-v2
        P4 = self.pos+v1-v2
        self.shape = Polygon([tuple(P1.reshape(1, -1)[0]),
                            tuple(P2.reshape(1, -1)[0]),
                            tuple(P3.reshape(1, -1)[0]),
                            tuple(P4.reshape(1, -1)[0])
                            ]) # Turn np vectors back to tuples and construct the shapely object
