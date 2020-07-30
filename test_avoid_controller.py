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
from genetic import GeneticAlgorithm

import neat


random.seed(5)

# Generate obstacles
course = Course()
# obstacles = course.default()
# obstacles = course.moreComplicated()
# obstacles = course.emptyCourse()
# obstacles = course.avoidCourse()
# obstacles = course.avoidCourse2()
obstacles = course.popcornCourse()

## Set up problem
folder_names = os.listdir('data')
folder_names.sort()
print(folder_names)
# Load up most recent Genetic Algorithm Run
with open('./data/'+folder_names[-1]+'/algorithm_pickle', 'rb') as f:
    genetic_alg = pickle.load(f)
drone, environment, stabiliser, preset = genetic_alg.decodeGenome(genetic_alg.readBestGenome())
print(genetic_alg.universal_best_generation)
environment.resetEnv(obstacles)

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



while not collision and total_t < 30.0:
    # start = time.time()
    delta_z_dot, delta_x_dot = drone.findDeltaVelocities()
    print("delta_z_dot: ", delta_z_dot)
    print("delta_x_dot: ", delta_x_dot)

    inputs_stabilise = [drone.vel[0][0]+delta_x_dot, drone.vel[1][0]+delta_z_dot, drone.theta_pos, drone.theta_vel] # inputs are laser lengths + state information
    # action = net.advance(inputs, preset.dt, preset.dt)
    action = stabiliser.activate(inputs_stabilise)

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
environment.fitness -= 8*drone.pos[1][0]

print(environment.fitness)
if draw_final_graph:
    data_stream.plotEnd(10)
