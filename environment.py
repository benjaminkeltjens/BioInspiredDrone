"""
This file describes the parent drone class, and child drone classes with different control systems

Benjamin Keltjens
July 2020
"""

import numpy as np
from drone import Drone
from shapely.geometry import Polygon, Point

class Environment(object):

    def __init__(self, lasers):
        self.obstacles = [] # List of obstacles objects, unordered
        self.obstacle_distances = [] # List of obstacle distances in same order as obstacles
        self.ordered_obstacles = [] # Ordered indexes of self.obstacles based on distance to Drone
        self.laser_angles = [None]*lasers
        self.laser_distances = [None]*lasers

    def update(self, drone):
        self.laser_angles = drone.laser_list()
        self.findLaserDistances(drone.pos)

    def orderObstacles(self, pos_drone):
        # Update and order the distancece of the obstacles

        # First update obstacle_distances
        for i in range(len(obstacles)):
            obstacle_distances[i] = obstacles[i].findDistance(pos_drone)

        self.ordered_obstacles = list(np.argsort(obstacle_distances)[::-1]) # Find the sorted of the distances, and reverse

    def findLaserDistances(self, pos_drone):

        for i in range(len(laser_angles)):
            # For each laser
            laser_m = -np.tan(laser_angles[i])
            laser_b = pos_drone[1][0] - laser_m * pos_drone[0][0]
            distance = np.sqrt(((-b/m)-pos_drone[0][0])**2 + (pos_drone[1][0])**2) # Find intersection with ground

            for j in ordered_obstacles:
                # For each obstacle find if laser collides with the determinant and then find intersections
                obstacle = self.obstacles[j]
                a = 1 - laser_m**2
                b = -(2*obstacle.pos[0][0] + 2*laser_m*(laser_b-obstacle.pos[1][0]))
                c = obstacle.pos[0][0]**2 - (laser_b-obstacle.pos[1][0]))**2 - obstacle.radius**2

                determinant = b**2 - 4*a*c
                if determinant < 0:
                    # There is no intersection, so continue search
                    continue
                elif determinant == 0:
                    # There is one intersection, however, this extremely unlikely
                    x_intersect = -b / (2*a)
                    z_intersect = laser_m*x_intersect+laser_b
                    distance = np.sqrt((x_intersect-pos_drone[0][0])**2 + (z_intersect-pos_drone[1][0])**2)
                    break # with the ordered list we can assume this is the closest object, to avoid analysing every obstacle
                else:
                    # Two intersections
                    x_1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
                    x_2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
                    z_1 = laser_m*x_1+laser_b
                    z_2 = laser_m*x_2+laser_b
                    d_1 = np.sqrt((x_1-pos_drone[0][0])**2 + (z_1-pos_drone[1][0])**2)
                    d_2 = np.sqrt((x_2-pos_drone[0][0])**2 + (z_2-pos_drone[1][0])**2)
                    distance = min(d_1, d_2)
                    break # with the ordered list we can assume this is the closest object, to avoid analysing every obstacle

            self.laser_distances[i] = distance

        def findCollision(self, drone):
            # Find if there are any collisions, return True or False

            obstacles_to_observe = [] # list with index of obstacles within maximum range for collision
            for j in self.ordered_obstacles:
                max_distance = 2*np.sqrt(drone.length**2 + drone.height**2) + obstacles[j].radius
                if obstacle_distances[j] <= max_distance:
                    obstacles_to_observe.append(j)

            if len(obstacles_to_observe) == 0:
                return False

            for j in obstacles_to_observe:
                #TODO: Find faster way find intersections
                if self.obstacles[j].shape.intersection(drone.shape) > 0:
                    # If there is an intersection between the two objects then there is a collision
                    return True

            return False


class Obstacle(object):

    def __init__(self,xposition,zposition,radius):
        self.pos = np.array([[xposition],[zposition]])
        self.radius = radius
        self.shape = Point(xpoistion,zposition).buffer(radius)

    def findDistance(self, pos_drone):
        # find distance between obstacle and drone
        return np.linalg.norm(self.pos - pos_drone)
