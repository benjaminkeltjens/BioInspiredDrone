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

from drone import TrainingDrone
from environment import Environment, Obstacle, Course
from presets import Presets

import neat

# Load Presets
preset = Presets()
preset.loadDefault()

def loadStabiliser(N):
    # Load stabiliser network

    local_dir = os.path.dirname(__file__)
    folder_paths = ['first_stabiliser', 'second_stabiliser', 'third_stabiliser', 'fourth_stabiliser', 'fifth_stabiliser']
    with open(folder_paths[N-1]+'/winner', 'rb') as f:
        stabiliser = pickle.load(f)
        config_path_stabilise = os.path.join(local_dir, folder_paths[N-1]+'/config-nn')
        config_stabilise = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path_stabilise)
        return neat.nn.FeedForwardNetwork.create(stabiliser, config_stabilise)

def loadStabiliserInfo(N):
    # Load Stabiliser network information

    local_dir = os.path.dirname(__file__)
    folder_paths = ['first_stabiliser', 'second_stabiliser', 'third_stabiliser', 'fourth_stabiliser', 'fifth_stabiliser']
    with open(folder_paths[N-1]+'/winner', 'rb') as f:
        stabiliser = pickle.load(f)
        config_path_stabilise = os.path.join(local_dir, folder_paths[N-1]+'/config-nn')
        config_stabilise = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path_stabilise)
        return stabiliser

def check_tolerance(vel, angular_vel, angle):
    # Check if drone in end state

    angle_tolerance = 5*np.pi/180
    vel_tolerance = 0.5
    angular_vel_tolerance = np.pi

    angle_flag = min(drone.theta_pos, abs(2*np.pi - drone.theta_pos)) <= angle_tolerance
    vel_flag = np.linalg.norm(vel) < vel_tolerance
    angular_vel_flag = abs(angular_vel) < angular_vel_tolerance

    return (angle_flag and vel_flag and angular_vel_flag)

def run_sim(drone, environment, net):
    #Simulation
    collision = False
    total_t = 0
    x_dis_max = 0.
    z_dis_max = 0.

    while total_t<20.0:
        # start = time.time()
        theta_raw = drone.theta_pos
        if theta_raw < np.pi:
            theta_input = theta_raw
        else:
            theta_input = np.pi - theta_raw
        inputs = [drone.vel[0][0], drone.vel[1][0], theta_input, drone.theta_vel]


        action = net.activate(inputs)

        drone.update(action[0]*drone.input_limit, action[1]*drone.input_limit)

        environment.update(drone, False)
        drone.recieveLaserDistances(environment.laser_distances)
        x_dis = abs(drone.pos[0][0]-preset.x_initial)
        z_dis = abs(drone.pos[1][0]-preset.z_initial)

        if x_dis > x_dis_max:
            x_dis_max = x_dis
        if z_dis > z_dis_max:
            z_dis_max = z_dis

        if check_tolerance(drone.vel, drone.theta_vel, drone.theta_pos):
            break
        # print(time.time()-start)
        total_t += preset.dt
        # print(total_t)

    return total_t, x_dis_max, z_dis_max
# Data Analysis Loop
course = Course()
obstacles = course.emptyCourse()
drone = TrainingDrone(preset.drone_dict)
avg_times = []
avg_x_lats = []
avg_z_lats = []

# Print controller information
print("PRINTING CONTROLLERS")
for i in range(5):
    print("Controller: " + str(i+1))
    print(loadStabiliserInfo(i+1))

# For each stabiliser  find the performance over 100 runs 
for i in range(5):
    net = loadStabiliser(i+1)
    random.seed(2)
    random.seed(4)
    times = []
    x_lat = []
    z_lat = []
    for j in range(100):
        print("i: " + str(i))
        print("j: " + str(j))
        # Find current obstacle Course
        preset.theta_intial = (random.random()*2-1)*25.*np.pi/180
        preset.createDroneDictionary()

        # Reset all object parameters for new run
        drone.resetParams(preset.drone_dict)
        # drone.vel = np.array([[(random.random()*2-1)*3], [(random.random()*2-1)*3]]) # [m/s]
        drone.theta_vel = (random.random()*2-1)*(np.pi/2)
        # Load and set presets

        environment = Environment(preset.lasers, obstacles, preset.max_laser_length, preset.safe_vel, preset.safe_angle)
        # Initialise None lists in environment and drone
        environment.update(drone, False)
        drone.recieveLaserDistances(environment.laser_distances)

        total_t, x_dis_max, z_dis_max = run_sim(drone, environment, net)
        if z_dis_max > 10:
            print("Here")

        times.append(total_t)
        x_lat.append(x_dis_max)
        z_lat.append(z_dis_max)

    avg_time = sum(times)/len(times)
    avg_x_lat = sum(x_lat)/len(x_lat)
    avg_z_lat = sum(z_lat)/len(z_lat)
    avg_times.append(avg_time)
    avg_x_lats.append(avg_x_lat)
    avg_z_lats.append(avg_z_lat)

fig = plt.figure()
plt.rcParams.update({'font.size': 15})
ax_time = fig.add_subplot(3,1,1)
ax_time.set_ylabel("Avg Settle Time [s]")
ax_x_lat = fig.add_subplot(3,1,2)
ax_x_lat.set_ylabel("Avg x displacement [m]")
ax_z_lat = fig.add_subplot(3,1,3)
ax_z_lat.set_ylabel("Avg z displacement [m]")
ax_z_lat.set_xlabel("Controller [-]")
ax_time.bar(list(range(1,6)),avg_times)
ax_x_lat.bar(list(range(1,6)),avg_x_lats)
ax_z_lat.bar(list(range(1,6)),avg_z_lats)
plt.show()








# net = neat.ctrnn.CTRNN.create(c, config, drone.dt)
