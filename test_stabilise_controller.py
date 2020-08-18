"""
This file is the test environment for the stabiliser controller

Benjamin Keltjens
July 2020
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import os

from drone import TrainingDrone
from environment import Environment, Obstacle, Course
from render import Renderer, DataStream
from presets import Presets

import neat

# Load Presets
preset = Presets()
preset.loadDefault()


# Initialise objects
drone = TrainingDrone(preset.drone_dict)

# Generate obstacles/load empty course
course = Course()
obstacles = course.emptyCourse()

environment = Environment(preset.lasers, obstacles, preset.max_laser_length, preset.safe_vel, preset.safe_angle)
# Initialise None lists in environment and drone
environment.update(drone, False)
drone.recieveLaserDistances(environment.laser_distances)

# Initialise Renderer and Plots
#Render Window limits
draw_scene = True # Flag to draw live simulation
draw_graphs = False # Flag to draw live data stream
draw_final_graph = True # Flag to show data at the end of simulation


xlims = [-11,11]
ylims = [0,40]
renderer = Renderer(obstacles, drone, xlims, ylims, draw_scene)
data_stream = DataStream(preset.input_limit, draw_graphs)

#Simulation
collision = False
total_t = 0

def loadStabiliser(N):
    # Load stabiliser network

    local_dir = os.path.dirname(__file__)
    folder_paths = ['first_stabiliser', 'second_stabiliser', 'third_stabiliser', 'fourth_stabiliser', 'fifth_stabiliser']
    with open(folder_paths[N-1]+'/winner', 'rb') as f:
        stabiliser = pickle.load(f)
    config_path_stabilise = os.path.join(local_dir,folder_paths[N-1]+'/config-nn')
    config_stabilise = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    config_path_stabilise)
    return neat.nn.FeedForwardNetwork.create(stabiliser, config_stabilise)

# Load stabilier network
net = loadStabiliser(3)

# Simulate dyanmics
while not collision and total_t<2.0:

    # Transform theta to positive-negative representation about positive x-axis
    theta_raw = drone.theta_pos
    if theta_raw < np.pi:
        theta_input = theta_raw
    else:
        theta_input = theta_raw - 2*np.pi

    inputs = [drone.vel[0][0], drone.vel[1][0], theta_input, drone.theta_vel]

    # Input inputs to network
    action = net.activate(inputs)

    # Update Dyanmics
    drone.update(action[0]*drone.input_limit, action[1]*drone.input_limit)

    # Update environment and renderes
    environment.update(drone, False)
    drone.recieveLaserDistances(environment.laser_distances)
    renderer.updateHistory(drone)
    if draw_scene:
        renderer.updateGraph(drone)
    data_stream.updateGraphs(drone, total_t)

    total_t += preset.dt
    print(total_t)
    collision = environment.collision or environment.touchdown

# Draw Final Graph
if draw_final_graph:
    renderer.drawLine()
    data_stream.plotEnd(10)
