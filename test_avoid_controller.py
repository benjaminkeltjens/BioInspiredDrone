"""
This file is the test environment for avoid controller

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
from genetic import GeneticAlgorithm

import neat

# Generate obstacle course
course = Course()
obstacles = course.avoidCourse()
# obstacles = course.popcornCourse(1.5)

## Set up problem
folder_names = os.listdir('data')
folder_names.sort()
# Load up most recent Genetic Algorithm Run
with open('./data/'+folder_names[-1]+'/algorithm_pickle', 'rb') as f:
    genetic_alg = pickle.load(f)
print(genetic_alg.mutation_variance)
# Decode Genome
drone, environment, stabiliser, preset = genetic_alg.decodeGenome(genetic_alg.readBestGenome())
print(genetic_alg.universal_best_generation)

# Reset environment
environment.resetEnv(obstacles)
preset.drone_dict["theta_intial"] = 0.0
preset.drone_dict["z_initial"] = 30.0
preset.drone_dict["x_initial"] = 0.0

# Reset Drone
drone.resetParams(preset.drone_dict)

# Initialise None lists in environment and drone
environment.update(drone, False)
drone.recieveLaserDistances(environment.laser_distances)

# Initialise Renderer and Plots
#Render Window limits
draw_scene = True # Flag to draw live simulation
draw_graphs = False # Flag to draw live data stream
draw_final_graph = True # Flag to show data at the end of simulation

# Set drawing limits
xlims = [-11,11]
ylims = [0,40]
# Load render and data stream
renderer = Renderer(obstacles, drone, xlims, ylims, draw_scene)
data_stream = DataStream(preset.input_limit, draw_graphs)

#Simulation
collision = False
total_t = 0
# Begin simulating scenario
while not collision and total_t < 30.0:

    # Find the delta velocities of the drone from the avoidance controller
    delta_z_dot, delta_x_dot = drone.findDeltaVelocities()
    print("delta_z_dot: ", delta_z_dot)
    print("delta_x_dot: ", delta_x_dot)
    print(drone.theta_pos)

    # Transform theta to positive-negative representation about positive x-axis
    theta_raw = drone.theta_pos
    if theta_raw < np.pi:
        theta_input = theta_raw
    else:
        theta_input = theta_raw - 2*np.pi

    # Change input stabiliser by the obstacle avoider output
    inputs_stabilise = [drone.vel[0][0]+delta_x_dot, drone.vel[1][0]+delta_z_dot, theta_input, drone.theta_vel] # inputs are laser lengths + state information

    # Give inputs to stabiliser network
    action = stabiliser.activate(inputs_stabilise)

    # Update dyanmics
    drone.update(action[0]*drone.input_limit, action[1]*drone.input_limit)

    # update environment and plotters
    environment.update(drone, False)
    drone.recieveLaserDistances(environment.laser_distances)
    renderer.updateHistory(drone)
    if draw_scene:
        renderer.updateGraph(drone)
    data_stream.updateGraphs(drone, total_t)

    total_t += preset.dt
    print(total_t)
    collision = environment.collision or environment.touchdown

# Draw final Graph
if draw_final_graph:
    renderer.drawLine()
    data_stream.plotEnd(10)
