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

from drone import Drone, SimpleLander
from environment import Environment, Obstacle, Course
from render import Renderer, DataStream
from presets import Presets

import neat

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
drone = SimpleLander(preset.drone_dict, 10, 1)

# Generate obstacles
course = Course()
# easy_obstacles = course.default()
# obstacles = course.moreComplicated()
obstacles = course.emptyCourse()

environment = Environment(preset.lasers, obstacles, preset.max_laser_length, preset.safe_vel, preset.safe_angle)
# Initialise None lists in environment and drone
environment.update(drone, False)
drone.recieveLaserDistances(environment.laser_distances)

# Initialise Renderer and Plots
#Render Window limits
draw_scene = False
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

def loadStabiliser(N):
    local_dir = os.path.dirname(__file__)
    folder_paths = ['first_stabiliser', 'second_stabiliser', 'third_aggressive_stabiliser', 'fourth_stabiliser', 'fifth_aggressive_stabiliser']
    with open('winner', 'rb') as f:
        stabiliser = pickle.load(f)
    config_path_stabilise = os.path.join(local_dir, 'config-nn')
    config_stabilise = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    config_path_stabilise)
    return neat.nn.FeedForwardNetwork.create(stabiliser, config_stabilise)

# net = neat.ctrnn.CTRNN.create(c, config, drone.dt)
net = loadStabiliser(1)

while not collision and total_t<20.0:
    # start = time.time()
    theta_raw = drone.theta_pos
    if theta_raw < np.pi:
        theta_input = theta_raw
    else:
        theta_input = theta_raw - 2*np.pi
    inputs = [drone.vel[0][0], drone.vel[1][0], theta_input, drone.theta_vel]

    # inputs =[drone.vel[0][0], drone.vel[1][0], drone.theta_pos, drone.theta_vel] # inputs are laser lengths + state information
    # action = net.advance(inputs, preset.dt, preset.dt)
    action = net.activate(inputs)

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
