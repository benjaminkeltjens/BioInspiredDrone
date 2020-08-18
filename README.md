## BioInspiredDrone
Python 3.6

# Requirements:
- neat-python
- matplotlib
- shapely
- datetime
- numpy
- os
- multiprocessing
- random
- graphviz

# This is the deliverable code for the Bio-Inspired Course
This code is comprised of various sections.

drone.py holds the drone classes need for simulating dynamics, lasers, and describing controllers

environment.py:
  - Environment class to simulate interaction of drone with obstacles and bounds in the Environment
  - Obstacle class
  - Course class for generating obstacles for the environment

presets.py loads the preset parameters for the drone and environment classes

render.py has all classes needed for visualising simulations, data stream and data analysis

# Stabiliser Controller Evolution:
 - run train_controller_stabilise.py to use implementation of NEAT algorithm to train drone to stabilise
 - run test_controller_stabilise.py to see drone stabilise simulation using chosen controllers
 - run plot_stabilisers.py to simulate and display performance of different stabilisation controllers
 - stabilise controllers are stored in stabiliser folders ('first_stabiliser', 'second_stabiliser', etc...)

# Avoid Controller Evolution:
- genetic.py has the class description of the entire genetic algorithm
- run train_controller_avoid.py to run implementation of genetic algorithm
- run test_controller_avoid.py to see simulation of avoid controllers
- run test_GA.py to find performance of avoidance controllers
- run plot_GA.py to see evolution of controllers
- data folder stores avoid controller evolution data from running Algorithm
- important_data stores history of evolution data you want analysed by test_GA and plot_GA
