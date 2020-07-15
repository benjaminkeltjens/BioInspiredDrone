"""
This file is the test environment for the control systems

Benjamin Keltjens
July 2020
"""

import numpy as np
import time

from drone import Drone
from environment import Environment, Obstacle
from render import Renderer

## Set up problem

# Drone charactertics
gravity = -9.80665 # [m/s^2]
mass = 5 # [kg]
length = 0.3 # [m]
height = 0.05 # [m]
lasers = 8
laser_range = 2*np.pi # [rad]
input_limit = 50 # [N]
dt = 0.01 #[s]

# Starting position
x_initial = 0. # [m]
z_initial = 10. # [m]


# Initialise objects
drone = Drone(x_initial, z_initial, gravity, mass, length, height, lasers, laser_range, input_limit, dt)

# Generate obstacles
obstacles = []
total_obstacles = 5
for i in range(total_obstacles):
    x = 0 - 4*(((total_obstacles-1)/2)-i)
    z = 5
    r = 0.5
    obstacles.append(Obstacle(x,z,r))
obstacles.append(Obstacle(0,13,0.5))

environment = Environment(lasers, obstacles)
# Initialise None lists in environment and drone
environment.update(drone)
drone.recieveLaserDistances(environment.laser_distances)

# Initialise Renderer
#Render Window limits
xlims = [-10,10]
ylims = [0,40]
renderer = Renderer(obstacles, drone, xlims, ylims)


# Simulation
collision = False

while not collision:
    input_L = mass*-gravity/2
    input_R = input_L
    drone.update(input_L, input_R)
    environment.update(drone)
    drone.recieveLaserDistances(environment.laser_distances)
    renderer.updateGraph(drone)

    collision = environment.collision
