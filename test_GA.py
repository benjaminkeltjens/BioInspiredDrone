from genetic import GeneticAlgorithm
from environment import Course
import random

random.seed(1)

course = Course()
course_1 = course.avoidCourse()
course_2 = course.avoidCourse2()
courses = [course_1,course_2]
algorithm = GeneticAlgorithm(0, 10, 24, 1.0, courses)
algorithm.run()
