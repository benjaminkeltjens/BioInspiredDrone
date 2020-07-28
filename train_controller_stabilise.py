"""
This file is the controller trainer

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
random.seed(1) #
# random.seed(2)
preset = Presets()
preset.loadDefault()

drone = TrainingDrone(preset.drone_dict)


global_run_count = 0
global_runs_per_net = 10
global_simulation_seconds = 10.0

def trainController(controller):


    preset.createDroneDictionary()
    drone.__init__(preset.drone_dict)

    # NN
    if controller == 1:
        preset.writeNNConfigFile()
        config_path = os.path.join(local_dir, 'config-nn')

    # CTRNN
    else:
        preset.writeCTRNNConfigFile()
        config_path = os.path.join(local_dir, 'config-ctrnn')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    if controller == 1:
        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_NN_genome)
    else:
        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_CTRNN_genome)

    winner = pop.run(pe.evaluate)
    # global_run_count = 0

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

def eval_CTRNN_genome(genome, config):
    net = neat.ctrnn.CTRNN.create(genome, config, drone.dt)
    fitnesses = []
    global_run_count = 0

    for _ in range(global_runs_per_net):
        # Find current obstacle Course
        if global_run_count%3 == 0:
            environment.resetEnv(course_1)
            preset.z_initial = 20.
            preset.x_initial = -2.
            preset.theta_intial = -10.*np.pi/180
        elif global_run_count%3 == 1:
            environment.resetEnv(course_1)
            preset.z_initial = 28.
            preset.x_initial = 1.
            preset.theta_intial = 8.*np.pi/180
        else:
            environment.resetEnv(course_1)
            preset.z_initial = 28.
            preset.x_initial = 1.
            preset.theta_intial = -16*np.pi/180


        preset.createDroneDictionary()

        # Reset all object parameters for new run
        drone.resetParams(preset.drone_dict)
        environment.update(drone, False)
        drone.recieveLaserDistances(environment.laser_distances)

        collision = False
        total_t = 0
        temp_fitness = 0
        while not collision:
            # create list of inputs
            inputs = [drone.vel[0][0], drone.vel[1][0], drone.theta_pos, drone.theta_vel] # inputs are laser lengths + state information
            action = net.advance(inputs, preset.dt, preset.dt)

            drone.update(action[0]*drone.input_limit, action[1]*drone.input_limit)

            # Here check if at the end of simulation (Not done in while statement to calculate the fitness correctly)
            if total_t < global_simulation_seconds and drone.pos[1][0] < 40. and abs(drone.pos[0][0]) < 10. :
                environment.update(drone, False)
            else:
                environment.update(drone, True)
                environment.fitness -= 10000
                break

            temp_fitness -= abs(drone.vel[1][0])
            temp_fitness -= abs(drone.vel[0][0])*0.5
            temp_fitness -= min(drone.theta_pos, abs(2*np.pi - drone.theta_pos))/(np.pi/9)
            temp_fitness -= abs(drone.theta_vel)/(2*np.pi/9)
            # temp_fitness -= (drone.input_L+drone.input_R)/50

            drone.recieveLaserDistances(environment.laser_distances)

            total_t += preset.dt
            collision = environment.collision or environment.touchdown
        global_run_count += 1
        fitnesses.append(temp_fitness/total_t)

    return min(fitnesses)

def eval_NN_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []
    global_run_count = 0
    random.seed(2)
    random.seed(1)
    for _ in range(global_runs_per_net):

        # Find current obstacle Course
        preset.theta_intial = (random.random()*2-1)*25.*np.pi/180
        preset.createDroneDictionary()

        # Reset all object parameters for new run
        drone.resetParams(preset.drone_dict)
        drone.vel = np.array([[(random.random()*2-1)*3], [(random.random()*2-1)*3]]) # [m/s]
        drone.theta_vel = (random.random()*2-1)*(np.pi/2)




        collision = False
        total_t = 0
        temp_fitness = 0
        while not collision:
            # create list of inputs
            inputs = [drone.vel[0][0], drone.vel[1][0], drone.theta_pos, drone.theta_vel] # inputs are laser lengths + state information
            action = net.activate(inputs)

            drone.updateStabilise(action[0]*drone.input_limit, action[1]*drone.input_limit)

            # Here check if at the end of simulation (Not done in while statement to calculate the fitness correctly)
            if not (total_t < global_simulation_seconds and drone.pos[1][0] < 40. and drone.pos[1][0] > 0. and abs(drone.pos[0][0]) < 2.5):
                if abs(drone.pos[0][0]) >= 2.5:
                    temp_fitness -= 100
                break

            temp_fitness -= abs(drone.vel[1][0])
            temp_fitness -= abs(drone.vel[0][0])
            temp_fitness -= min(drone.theta_pos, abs(2*np.pi - drone.theta_pos))/(np.pi/9)
            # temp_fitness -= abs(drone.theta_vel)/(2*np.pi/9)
            # temp_fitness -= (drone.input_L+drone.input_R)/50

            total_t += preset.dt
        global_run_count += 1
        fitnesses.append(temp_fitness/total_t)

    return min(fitnesses)





if __name__ == '__main__':
    trainController(1)
