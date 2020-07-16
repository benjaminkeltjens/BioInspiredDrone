"""
This file is the test environment for the control systems

Benjamin Keltjens
July 2020
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from drone import Drone, SimpleLander
from environment import Environment, Obstacle
from render import Renderer, DataStream
from presets import Presets

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
obstacles = []
total_obstacles = 4
for i in range(total_obstacles):
    x = 0 - 4*(((total_obstacles-1)/2)-i)
    z = 5
    r = 0.5
    obstacles.append(Obstacle(x,z,r))
# obstacles.append(Obstacle(0,13,0.5))

environment = Environment(preset.lasers, obstacles, preset.max_laser_length)
# Initialise None lists in environment and drone
environment.update(drone)
drone.recieveLaserDistances(environment.laser_distances)

# Initialise Renderer and Plots
#Render Window limits
draw_scene = True
draw_graphs = False
draw_final_graph = True

if draw_scene:
    xlims = [-10,10]
    ylims = [0,20]
    renderer = Renderer(obstacles, drone, xlims, ylims)
data_stream = DataStream(preset.input_limit, draw_graphs)

#Simulation
collision = False
total_t = 0
while not collision:
    start = time.time()

    input_L, input_R = drone.findInput()
    drone.update(input_L, input_R)
    environment.update(drone)
    drone.recieveLaserDistances(environment.laser_distances)
    if draw_scene:
        renderer.updateGraph(drone)
    data_stream.updateGraphs(drone, total_t)

    print(time.time()-start)
    total_t += preset.dt
    collision = environment.collision

if draw_final_graph:
    data_stream.plotEnd(5)
