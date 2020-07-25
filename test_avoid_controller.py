"""
This file is the test environment for the control systems

Benjamin Keltjens
July 2020
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import os
import random

from drone import Drone, AvoiderDrone
from environment import Environment, Obstacle, Course
from render import Renderer, DataStream
from presets import Presets

import neat

random.seed(127)

## Set up problem

# Drone charactertics
# gravity = -9.80665 # [m/s^2]
# mass = 5 # [kg]
# length = 0.3 # [m]
# height = 0.05 # [m]
# lasers = 10
# laser_range = 2*np.pi # [rad]
# input_limit = 50 # [N]
# dt = 0.01 #[s]
# max_laser_length = 10
#
# # Starting position
# x_initial = -0.8 # [m]
# z_initial = 10. # [m]
preset = Presets()
preset.loadDefault()


# Initialise objects
# drone = Drone(x_initial, z_initial, gravity, mass, length, height, lasers, laser_range, input_limit, dt)
drone = AvoiderDrone(preset.drone_dict, 0.0, 0.0, 5., 5., 0.4, 0.4, 0., 0.6, 3.0)

# Generate obstacles
course = Course()
obstacles = course.default()
# obstacles = course.moreComplicated()
# obstacles = course.avoidCourse()
# obstacles = course.avoidCourse2()
# obstacles = course.emptyCourse()

environment = Environment(preset.lasers, obstacles, preset.max_laser_length, preset.safe_vel, preset.safe_angle)
# Initialise None lists in environment and drone
environment.update(drone, False)
drone.recieveLaserDistances(environment.laser_distances)

# Initialise Renderer and Plots
#Render Window limits
draw_scene = True
draw_graphs = False
draw_final_graph = True

if draw_scene:
    xlims = [-11,11]
    ylims = [0,40]
    renderer = Renderer(obstacles, drone, xlims, ylims)
data_stream = DataStream(preset.input_limit, draw_graphs)

#Simulation
collision = False
total_t = 0

# Load controller
with open('stabilise_controller', 'rb') as f:
    stabiliser = pickle.load(f)

local_dir = os.path.dirname(__file__)
# config_path = os.path.join(local_dir, 'config-ctrnn')
config_path_stabilise = os.path.join(local_dir, 'config-stabilise')
config_stabilise = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path_stabilise)

# net = neat.ctrnn.CTRNN.create(c, config, drone.dt)
net_stabilise = neat.nn.FeedForwardNetwork.create(stabiliser, config_stabilise)

while not collision and total_t < 30.0:
    # start = time.time()
    delta_z_dot, delta_x_dot, delta_theta_dot = drone.findDeltaVelocities()
    print("delta_z_dot: ", delta_z_dot)
    print("delta_x_dot: ", delta_x_dot)

    inputs_stabilise = [drone.vel[0][0]+delta_x_dot, drone.vel[1][0]+delta_z_dot, drone.theta_pos, drone.theta_vel+delta_theta_dot] # inputs are laser lengths + state information
    # action = net.advance(inputs, preset.dt, preset.dt)
    action = net_stabilise.activate(inputs_stabilise)

    drone.update(action[0]*drone.input_limit, action[1]*drone.input_limit)

    environment.update(drone, False)
    drone.recieveLaserDistances(environment.laser_distances)
    if draw_scene:
        renderer.updateGraph(drone)
    data_stream.updateGraphs(drone, total_t)

    # print(time.time()-start)
    total_t += preset.dt
    print(total_t)
    collision = environment.collision or environment.touchdown
    print(environment.fitness)

print(environment.fitness)
if draw_final_graph:
    data_stream.plotEnd(10)
