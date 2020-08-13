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
course_1 = course.popcornCourse(1.5)
course_2 = course.popcornCourse(2.5)
course_3 = course.popcornCourse(5.5)
courses = [course_1, course_2, course_3]

# Algorithm Paramaters
threshold = 0.
generation_limit =  30
population_size = 48 # Divisible by 4
mutation_variance = 1.25

algorithm = GeneticAlgorithm(threshold, generation_limit, population_size, mutation_variance, courses)
algorithm.run()
