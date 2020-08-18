"""
This file is to run to avoid controller trainer

Benjamin Keltjens
July 2020
"""

import numpy as np
import time
import os
import pickle
import random

from genetic import GeneticAlgorithm
from environment import Course

import neat


# Generate obstacles
course = Course()
course_1 = course.popcornCourse(1.5)
course_2 = course.popcornCourse(11.5)
course_3 = course.avoidCourse()
courses = [course_1, course_2, course_3]

random.seed(1)
# Algorithm Paramaters
threshold = 0. # Fitness Threshold
generation_limit =  30
population_size = 48 # Divisible by 4
mutation_variance = 1.5

# Load Algorithm Object
algorithm = GeneticAlgorithm(threshold, generation_limit, population_size, mutation_variance, courses)
# Run Genetic Algorithm
algorithm.run()
