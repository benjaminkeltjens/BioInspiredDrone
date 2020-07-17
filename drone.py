"""
This file describes the parent drone class, and child drone classes with different control systems

Benjamin Keltjens
July 2020
"""
import numpy as np
from shapely.geometry import Polygon

class Drone(object):
    # The parent drone describes the kinematics and dynamics of the drone, handles inputs

    def __init__(self, xposition, zposition, theta, gravity, mass, length, height, lasers, laser_width, input_limit, input_rate_limit, dt):
        # Describe the drone charactertics

        self.mass = mass # [kg]
        self.length = length # [m]
        self.height = height # [m]
        self.inertia = (1/12)*mass*(height**2 + length**2) # [kg m^2] Approximated as rectangle to rotate
        self.lasers = lasers # number of lasers
        assert(laser_width <= np.pi*2 and laser_width > 0.0) # Make sure that laser range falls in realistic bounds
        self.laser_width = laser_width # [rad]
        self.laser_list = [None] * lasers # [rad]
        self.laser_distances = [None] * lasers # [m]
        self.input_limit = input_limit # [N]
        self.input_rate_limit = input_rate_limit # [N/s]
        self.dt = dt # [s]
        self.gravity = gravity # [m/s^2]

        # Initialise the drone static at a given location

        self.pos = np.array([[xposition], [zposition]]) # x,z [m]
        self.theta_pos = theta # [rad]
        self.vel = np.array([[0.], [0.]]) # [m/s]
        self.total_vel = np.linalg.norm(self.vel)
        self.theta_vel = 0. # [rad/s]
        self.accel = np.array([[0.], [gravity]]) # [m/s^2]
        self.theta_accel = 0. # [rad/s^2]
        self.input_L = self.mass*-self.gravity/2 # [N] start at hover
        self.input_R = self.mass*-self.gravity/2 # [N]
        self.shape = None
        self.updateShape()
        self.updateLaserAngles()



    def update(self, input_L, input_R):
        # Find dynamics, kinematics, and update states

        # Limit inputs and save the to history
        input_L = self.limitInput(input_L, self.input_L)
        input_R = self.limitInput(input_R, self.input_R)
        self.input_L = input_L
        self.input_R = input_R

        # Calculate dynamics
        forces, moment = self.resolveDynamics(input_L, input_R) # [N], [N m]
        self.accel = forces/self.mass
        self.theta_accel = moment/self.inertia

        # Euler Integration
        self.vel += self.accel * self.dt
        self.total_vel = np.linalg.norm(self.vel)
        self.theta_vel += self.theta_accel * self.dt

        self.pos += self.vel * self.dt
        self.theta_pos += self.theta_vel * self.dt
        self.theta_pos = self.wrapAngle(self.theta_pos, 2*np.pi) # wrap angle to stay between bounds of 0 and 2pi

        # Update the shapely representation of the drone and the laser angles
        self.updateShape()
        self.updateLaserAngles()

    def findInput(self):
        # This is the function that calculates the input to the two motors. It is more a dummy parent function that should be overwritten by the child classes
        # This dummy controller just hovers.
        input_L = mass*-gravity/2
        input_R = input_L
        return input_L, input_R

    def recieveLaserDistances(self, laser_distances):
        # This is a seperate function for inputing the laser distances from the environment object
        self.laser_distances = laser_distances

    def resolveDynamics(self, input_L, input_R):
        # Find linear and rotational dynamic from input

        body_thrust = input_L + input_R # Inputs as [N]
        global_thrust = np.array([[body_thrust*np.sin(self.theta_pos)], [body_thrust*np.cos(self.theta_pos)]])
        global_forces = global_thrust + np.array([[0], [self.gravity*self.mass]])

        moment = self.length * (input_L - input_R) * 0.5

        return global_forces, moment

    def updateLaserAngles(self):
        # Update list of the laser angles ordered in the negative theta direction
        if self.laser_width != np.pi*2:
            for i in range(self.lasers):
                laser_angle = self.theta_pos + (np.pi/2) + (self.laser_width/(self.lasers-1))*(((self.lasers-1)/2)-i)
                self.laser_list[i] = self.wrapAngle(laser_angle, np.pi*2)
        else:
            for i in range(self.lasers):
                laser_angle = self.theta_pos + (np.pi/2) + (self.laser_width/self.lasers)*((self.lasers/2)-i)
                self.laser_list[i] = self.wrapAngle(laser_angle, np.pi*2)

    def wrapAngle(self, angle, maximum):
        # Wrap angle between 0 and maximum
        return (angle%maximum)

    def updateShape(self):
        # Update drone shape representation

        # Create vectors to go from centre location to four corners
        v1 = np.array([[-(self.length/2)*np.cos(self.theta_pos)],[(self.length/2)*np.sin(self.theta_pos)]])
        v2 = np.array([[(self.height/2)*np.sin(self.theta_pos)],[(self.height/2)*np.cos(self.theta_pos)]])

        P1 = self.pos+v1+v2
        P2 = self.pos-v1+v2
        P3 = self.pos-v1-v2
        P4 = self.pos+v1-v2
        self.shape = Polygon([tuple(P1.reshape(1, -1)[0]),
                            tuple(P2.reshape(1, -1)[0]),
                            tuple(P3.reshape(1, -1)[0]),
                            tuple(P4.reshape(1, -1)[0])
                            ]) # Turn np vectors back to tuples and construct the shapely object

        self.xcoords, self.zcoords = self.shape.exterior.xy # create the x and y coords for plotting

    def limitInput(self, input, previous_input):
        # find the max and min possible inputs
        max_rate = previous_input + self.input_rate_limit*self.dt
        min_rate = previous_input - self.input_rate_limit*self.dt

        if input < max(0, min_rate):
            return max(0, min_rate)
        if input > min(self.input_limit, max_rate):
            return min(self.input_limit, max_rate)

        return input

class CTRNN(Drone):

    def __init__(self, drone_dict):


        # Initialise Drone parent class
        super().__init__(drone_dict["x_initial"], drone_dict["z_initial"], drone_dict["gravity"], drone_dict["mass"],
        drone_dict["length"], drone_dict["height"], drone_dict["lasers"], drone_dict["laser_range"], drone_dict["input_limit"],
        drone_dict["input_rate_limit"], drone_dict["dt"])

class SimpleLander(Drone):

    def __init__(self, drone_dict, gain_p, gain_i):
        self.error_i = 0
        self.gain_p = gain_p
        self.gain_i = gain_i
        self.land_vel = 0.6

        # Initialise Drone parent class
        super().__init__(drone_dict["x_initial"], drone_dict["z_initial"], drone_dict["theta_intial"], drone_dict["gravity"],
        drone_dict["mass"], drone_dict["length"], drone_dict["height"], drone_dict["lasers"], drone_dict["laser_range"],
        drone_dict["input_limit"], drone_dict["input_rate_limit"], drone_dict["dt"])

    def findInput(self):
        # max_vel = np.sqrt(self.land_vel**2 - 2*(2*self.input_limit/self.mass)*self.pos[1][0])
        target_z_dot = -min((0.1*self.pos[1][0]**2+self.land_vel),10)
        error = target_z_dot - self.vel[1][0]
        T = self.gain_p*error + self.gain_i*self.error_i
        self.error_i += error
        if abs(self.error_i*self.gain_i) > 50:
            self.error_i = (self.error_i/self.error_i)*abs(50/self.gain_i)
        input_L = T/2
        input_R = input_L
        return input_L, input_R
