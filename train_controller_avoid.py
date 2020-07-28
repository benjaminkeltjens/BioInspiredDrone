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


random.seed(5)

# Generate obstacles
course = Course()
course_1 = course.avoidCourse()
course_2 = course.avoidCourse2()
courses = [course_1, course_2]

# Algorithm Paramaters
threshold = -30.
generation_limit = 25
population_size = 48 # Divisible by 4
mutation_variance = 2.0

algorithm = GeneticAlgorithm(threshold, generation_limit, population_size, mutation_variance, courses)
algorithm.run()
