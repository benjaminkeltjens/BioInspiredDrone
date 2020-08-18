"""
This file is the test environment for testing the GeneticAlgorithm results

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

folder_names = os.listdir('important_data')
print(folder_names)
folder_names.sort()
print(folder_names)


#-----------------------------------------------------------------------------------

angle_ranges = []
for i in range(12):
    angle_ranges.append((np.pi/6)+i*(np.pi/6))

parameter_limits = [[0,3], #z_sat
    [0,3], #x_sat
    [1,10], #z_lim
    [1,10], #x_lim
    [0,3], #z_norm
    [0,3], #x_norm
    [0,1], #z_up
    [2,5], #max_vel
    [0,len(angle_ranges)-1], #angle range choice (index)
    [2,20], #number of lasers
    [1,5]] #stabiliser_choice

parameter_ranges = []
for i in range(len(parameter_limits)):
    parameter_ranges.append(parameter_limits[i][1]-parameter_limits[i][0])

best_genomes = []
for name in folder_names:
    genome = list(np.loadtxt('/home/benjamin/git/BioInspiredDrone/important_data/'+name+'/best_genome.txt'))
    genome[8] = int(genome[8]); genome[9] = int(genome[9]); genome[10] = int(genome[10])
    best_genomes.append(genome)

labels = ['z_sat', 'x_sat', 'z_lim', 'x_lim', 'z_norm', 'x_norm', 'z_up', 'max_vel', 'angle_range', 'number of lasers', 'stabiliser']

for i in range(len(parameter_limits)):
    print(labels[i])
    values = []
    for j in range(len(best_genomes)):
        print(folder_names[j] + ': ' + str(best_genomes[j][i]))
        if labels[i] == 'angle_range':
            values.append(angle_ranges[best_genomes[j][i]])
        else:
            values.append(best_genomes[j][i])
    print('Average = ' + str(np.mean(values)))
    print('Std Dev = ' + str(np.std(values)))
    print('Normalise Std Dev = ' + str(np.std(values)/parameter_ranges[i]))
    print('--------------------------------------')

#-----------------------------------------------------------------------------------
courses = []
course = Course()
for i in range(100):
    courses.append(course.popcornCourse(i))

threshold = 0.
generation_limit =  30
population_size = 48 # Divisible by 4
mutation_variance = 1.5

algorithm = GeneticAlgorithm(threshold, generation_limit, population_size, mutation_variance, [courses[0]])

def testOneGenome(genome,courses,algorithm):
    drone, environment, stabiliser, preset_copy = algorithm.decodeGenome(genome)
    # For each obstacle course
    counter = 0
    energy = 0
    for c in range(len(courses)):

        # Reset all object parameters for new run (if not first run)
        environment.resetEnv(courses[c])
        drone.resetParams(preset_copy.drone_dict)
        environment.update(drone, False)
        drone.recieveLaserDistances(environment.laser_distances)

        collision = False
        total_t = 0
        while not collision and total_t < 30.0:

            delta_z_dot, delta_x_dot = drone.findDeltaVelocities()
            # create list of inputs

            theta_raw = drone.theta_pos
            if theta_raw < np.pi:
                theta_input = theta_raw
            else:
                theta_input = theta_raw - 2*np.pi

            inputs_stabilise = [drone.vel[0][0]+delta_x_dot, drone.vel[1][0]+delta_z_dot, theta_input, drone.theta_vel] # inputs are laser lengths + state information
            # action = net.advance(inputs, preset.dt, preset.dt)
            action = stabiliser.activate(inputs_stabilise)

            drone.update(action[0]*drone.input_limit, action[1]*drone.input_limit)


            environment.update(drone, False)
            drone.recieveLaserDistances(environment.laser_distances)

            total_t += preset_copy.dt
            collision = environment.collision or environment.touchdown
        counter += environment.touchdown
        if counter:
            energy += environment.energy

    success_rate = counter/len(courses)
    avg_energy = energy/counter

    return counter/len(courses), avg_energy

success_rates = []
for i in range(len(best_genomes)):
    start = time.time()
    print('Genome for: ' + folder_names[i])
    success_rate,energy = testOneGenome(best_genomes[i],courses,algorithm)
    success_rates.append(success_rate)
    print('Success Rate: ' + str(success_rate))
    print('Energy: ' + str(energy))
    print('Time Taken: ' + str(time.time()-start))
    print('--------------------------------------')
