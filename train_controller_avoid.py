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

    # Load  Stabiliser controller
with open('stabilise_controller', 'rb') as f:
    stabiliser = pickle.load(f)

# config_path = os.path.join(local_dir, 'config-ctrnn')
config_path_stabilise = os.path.join(local_dir, 'config-stabilise')
config_stabilise = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
neat.DefaultSpeciesSet, neat.DefaultStagnation,
config_path_stabilise)

net_stabilise = neat.nn.FeedForwardNetwork.create(stabiliser, config_stabilise)

# Load Default Preset
random.seed(5) # This one results in landing 13 seconds
# random.seed(2)
preset = Presets()
preset.loadDefault()
drone = TrainingDrone(preset.drone_dict)

# Generate obstacles
course = Course()
# course_1 = course.moreComplicated()
course_1 = course.avoidCourse()
course_2 = course.avoidCourse2()

environment = Environment(preset.lasers, course_1, preset.max_laser_length, preset.safe_vel, preset.safe_angle)

global_run_count = 0
global_runs_per_net = 2
global_simulation_seconds = 30.0

def trainController(N_lasers, laser_range, max_laser_length, controller):

    # Load variables and set number of lasers
    preset.lasers = N_lasers
    preset.laser_range = laser_range
    preset.max_laser_length = max_laser_length
    preset.createDroneDictionary()
    drone.__init__(preset.drone_dict)
    environment.__init__(preset.lasers, course_1, preset.max_laser_length, preset.safe_vel, preset.safe_angle)



if __name__ == '__main__':
    trainController(10, 1*np.pi, 5, 1)
