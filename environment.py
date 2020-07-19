"""
This file describes the parent drone class, and child drone classes with different control systems

Benjamin Keltjens
July 2020
"""

import numpy as np
from drone import Drone
from shapely.geometry import Point

class Environment(object):

    def __init__(self, lasers, obstacles, max_laser_length, safe_vel, safe_angle):
        self.obstacles = obstacles # List of obstacles objects, unordered
        self.obstacle_distances = [None]*len(obstacles) # List of obstacle distances in same order as obstacles
        self.ordered_obstacles = [None]*len(obstacles) # Ordered indexes of self.obstacles based on distance to Drone
        self.laser_angles = [None]*lasers
        self.laser_distances = [None]*lasers
        self.max_laser_length = max_laser_length
        self.safe_vel = safe_vel
        self.safe_angle = safe_angle
        self.collision = False
        self.touchdown = False
        self.safe_touchdown = False
        self.fitness = 0

    def update(self, drone, end):
        self.laser_angles = drone.laser_list
        self.orderObstacles(drone.pos)
        self.findLaserDistances(drone.pos)
        self.collision = self.findCollision(drone)
        self.touchdown = self.checkTouchdown(drone)
        self.safe_touchdown = self.checkSafeTouchdown(drone)
        self.updateControllerFitness(drone,end)

    def orderObstacles(self, pos_drone):
        # Update and order the distancece of the obstacles

        # First update obstacle_distances
        for i in range(len(self.obstacles)):
            self.obstacle_distances[i] = self.obstacles[i].findDistance(pos_drone)
        self.ordered_obstacles = list(np.argsort(self.obstacle_distances)) # Find the sorted of the distances, and reverse

    def findLaserDistances(self, pos_drone):

        for i in range(len(self.laser_angles)):
            # For each laser
            laser_m = -np.tan(self.laser_angles[i])
            if abs(laser_m) > 10000: # limit laser slope to not have floating point errors in calculations
                laser_m = 10000 # Sign doesn't matter for slope intersection
            laser_b = pos_drone[1][0] - laser_m * pos_drone[0][0]

            # Initialise distance to maximum
            if self.laser_angles[i] > 0 and self.laser_angles[i] < np.pi:
                # If laser is facing ground
                distance = np.sqrt(((-laser_b/laser_m)-pos_drone[0][0])**2 + (pos_drone[1][0])**2) # Find intersection with ground [m]
                if distance > self.max_laser_length:
                    distance = self.max_laser_length
            else:
                # If laser doesn't face grond then laser goes to infinity
                distance = self.max_laser_length # TODO: Find out if this is wrong

            laser_vector = np.array([np.cos(self.laser_angles[i]), -np.sin(self.laser_angles[i])]) # Calculate once here to avoid Calculating multiple times

            for j in self.ordered_obstacles:
                # For each obstacle find if laser collides with the determinant and then find intersections
                obstacle = self.obstacles[j]
                a = 1 + laser_m**2
                b = (-2*obstacle.pos[0][0] + 2*laser_m*(laser_b-obstacle.pos[1][0]))
                c = obstacle.pos[0][0]**2 + (laser_b-obstacle.pos[1][0])**2 - obstacle.radius**2

                determinant = b**2 - 4*a*c
                if determinant < 0:
                    # There is no intersection, so continue search
                    continue

                # Check that the obstacle is in the correct direction with dot product between relative position of the
                # drone and obstacle and the laser vector more than 0
                relative_position = obstacle.pos-pos_drone
                if laser_vector.dot(relative_position.flatten()) < 0:
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

            self.laser_distances[i] = min(distance,self.max_laser_length)

    def findCollision(self, drone):
        # Find if there are any collisions, return True or False

        obstacles_to_observe = [] # list with index of obstacles within maximum range for collision
        for j in self.ordered_obstacles:
            max_distance = 2*np.sqrt(drone.length**2 + drone.height**2) + self.obstacles[j].radius
            if self.obstacle_distances[j] <= max_distance:
                obstacles_to_observe.append(j)

        if len(obstacles_to_observe) == 0:
            # If there are no objects in the maximum range then return no collisions
            return False

        for j in obstacles_to_observe:
            #TODO: Find faster way to find intersections
            if self.obstacles[j].shape.intersection(drone.shape).area > 0:
                # If there is an intersection between the two objects then there is a collision
                return True

        return False

    def checkTouchdown(self, drone):
        if drone.pos[1][0]-(drone.height/2) < 0:
            return True
        return False

    def checkSafeTouchdown(self, drone):
        vel_cond = drone.total_vel <= self.safe_vel # Lower than landing velocity (norm velocity)
        angle_cond = drone.theta_pos <= self.safe_angle or 2*np.pi-drone.theta_pos <= self.safe_angle # Lower than landing angle
        if (drone.pos[1][0]-(drone.height/2) < 0) and vel_cond and angle_cond:
            return True
        return False

    def updateControllerFitness(self, drone, end):
        if self.collision: # if there is a collision with an obstacle
            self.fitness -= 500
            # self.fitness -= 500 * drone.total_vel
        if end: # if there is no landing by the end of the run
            self.fitness -= 100

        if self.touchdown and not self.safe_touchdown: # If touchdown in unsafe manner
            self.fitness -= 400

            # if drone.theta_pos > np.pi:
            #     angle_error = (2*np.pi-drone.theta_pos) - drone.safe_angle
            # else:
            #     angle_error = drone.theta_pos - drone.safe_angle
            #
            # self.fitness -= 400 * (abs(drone.safe_vel - drone.total_vel) + abs(angle_error))

        self.fitness -= (drone.dt/120)*(drone.input_L + drone.input_R)
        self.fitness -= (drone.dt/60)*drone.lasers

class Obstacle(object):

    def __init__(self,xposition,zposition,radius):
        self.pos = np.array([[xposition],[zposition]])
        self.radius = radius
        self.shape = Point(xposition,zposition).buffer(radius)
        self.xcoords, self.zcoords = self.shape.exterior.xy # create the x and y coords for plotting

    def findDistance(self, pos_drone):
        # find distance between obstacle and drone
        return np.linalg.norm(self.pos - pos_drone)
