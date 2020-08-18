"""
This file is to train the stabilier controller

Benjamin Keltjens
July 2020
"""

import numpy as np
import time
import multiprocessing
import os
import pickle
import random

from drone import TrainingDrone
from environment import Environment, Obstacle, Course
from presets import Presets

import neat
import visualize

local_dir = os.path.dirname(__file__)

# Load Default Preset
random.seed(1)
preset = Presets()
preset.loadDefault()

# Load Training Drone
drone = TrainingDrone(preset.drone_dict)


# Set global parameters
global_run_count = 0
global_runs_per_net = 10
global_simulation_seconds = 10.0

def trainController(controller):
    # Function to setup and run NEAT genetic algorithm for stabiliser controller

    # Load presets
    preset.createDroneDictionary()
    drone.__init__(preset.drone_dict)

    # Create Neural Network config
    if controller == 1:
        preset.writeNNConfigFile()
        config_path = os.path.join(local_dir, 'config-nn')

    # Load neat config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Setup neat objects
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    if controller == 1:
        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_NN_genome)

    # Run NEAT algorithm
    winner = pop.run(pe.evaluate, n=50)

    # Display winner
    with open('winner', 'wb') as f:
        pickle.dump(winner, f)
    print("Printing Winner ====================================================")
    print(winner)

    visualize.plot_stats(stats, ylog=True, view=True, filename="ctrnn-fitness.svg")
    visualize.plot_species(stats, view=True, filename="ctrnn-speciation.svg")

    node_names = {-1: 'dx', -2: 'dz', -3: 'theta', -4: 'dtheta'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-ctrnn.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-ctrnn-enabled.gv", show_disabled=False)
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-ctrnn-enabled-pruned.gv", show_disabled=False, prune_unused=True)

def eval_NN_genome(genome, config):
    # This is the function to describe how to evaluate a genome and return fitness for genome

    # Generate stabilier
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []
    global_run_count = 0
    random.seed(2)
    random.seed(1)

    for _ in range(global_runs_per_net):

        # Find current intial conditions
        preset.theta_intial = (random.random()*2-1)*25.*np.pi/180
        preset.createDroneDictionary()

        # Reset all object parameters for new run
        drone.resetParams(preset.drone_dict)
        drone.vel = np.array([[(random.random()*2-1)*3], [(random.random()*2-1)*3]]) # [m/s]
        drone.theta_vel = (random.random()*2-1)*(np.pi/2)

        collision = False
        total_t = 0
        temp_fitness = 0

        # Simulate dynamics
        while not collision:

            # Transform theta to positive-negative representation about positive x-axis
            theta_raw = drone.theta_pos
            if theta_raw < np.pi:
                theta_input = theta_raw
            else:
                theta_input = theta_raw - 2*np.pi

            # create list of inputs
            inputs = [drone.vel[0][0], drone.vel[1][0], theta_input, drone.theta_vel] # inputs are laser lengths + state information

            # Input inputs to network
            action = net.activate(inputs)

            # Update dyanmics
            drone.updateStabilise(action[0]*drone.input_limit, action[1]*drone.input_limit)

            # Here check if at the end of simulation (Not done in while statement to calculate the fitness correctly)
            bound = 10.0
            if not (total_t < global_simulation_seconds and drone.pos[1][0] < 40. and drone.pos[1][0] > 0. and abs(drone.pos[0][0]) < bound):
                if abs(drone.pos[0][0]) >= bound:
                    temp_fitness -= 100
                break

            # Calculate fitness
            temp_fitness -= 1.0*abs(drone.vel[1][0])
            temp_fitness -= 1.0*abs(drone.vel[0][0])
            temp_fitness -= min(drone.theta_pos, abs(2*np.pi - drone.theta_pos))/(np.pi/9)
            temp_fitness -= abs(drone.theta_vel)/(2*np.pi/9)
            # temp_fitness -= (drone.input_L+drone.input_R)/50

            # Update Time
            total_t += preset.dt
        # Increase run count
        global_run_count += 1
        # add to list of fitness
        fitnesses.append(temp_fitness/total_t)

    # Return minimum fitness
    return min(fitnesses)

if __name__ == '__main__':
    trainController(1)
