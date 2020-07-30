"""
This file is the controller trainer

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
course_1 = course.avoidCourse()
course_2 = course.avoidCourse2()
course_3 = course.popcornCourse()
courses = [course_1, course_2, course_3]

seed = 7.
random.seed(seed)

# Algorithm Paramaters
threshold = 0.
generation_limit =  200
population_size = 48 # Divisible by 4
mutation_variance = 1.0

algorithm = GeneticAlgorithm(threshold, generation_limit, population_size, mutation_variance, courses, seed)
algorithm.run()
