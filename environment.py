"""
This file describes the environment, obstacle and course classes

Benjamin Keltjens
July 2020
"""

import numpy as np
from drone import Drone
from shapely.geometry import Point
import random

class Environment(object):

    def __init__(self, lasers, obstacles, max_laser_length, safe_vel, safe_angle):
        # Initialise environment objects

        self.obstacles = obstacles # List of obstacles objects, unordered
        self.obstacle_distances = [None]*len(obstacles) # List of obstacle distances in same order as obstacles
        self.ordered_obstacles = [None]*len(obstacles) # Ordered indexes of self.obstacles based on distance to Drone
        self.lasers = lasers # Number of lasers
        self.laser_angles = [None]*lasers # List Laser angles
        self.laser_distances = [None]*lasers # Distance of each laser
        self.max_laser_length = max_laser_length
        self.safe_vel = safe_vel # Safe Landing Velocity
        self.safe_angle = safe_angle # Safe angular position for landing
        self.collision = False # Collision Flag
        self.touchdown = False # Touchdown Flag
        self.safe_touchdown = False # Safe touchdown flag
        self.fitness = 0 # Total fitness for run
        self.energy = 0 # Total energy for run
        self.x_wall = 10 # Location of the walls in the x direction

    def resetEnv(self, obstacles):
        # Reset environment values

        self.obstacles = obstacles
        self.obstacle_distances = [None]*len(self.obstacles)
        self.ordered_obstacles = [None]*len(self.obstacles)
        self.laser_angles = [None]*self.lasers
        self.laser_distances = [None]*self.lasers
        self.collision = False
        self.touchdown = False
        self.safe_touchdown = False
        self.fitness = 0
        self.energy = 0

    def update(self, drone, end):
        # Update the state of the environment

        self.laser_angles = drone.laser_list # Get laser list from drone object
        self.orderObstacles(drone.pos) # Order the obstacles based on distance
        self.findLaserDistances(drone.pos) # Find the distance of each laser sensor
        self.collision = self.findCollision(drone) # Check for collisions
        self.touchdown = self.checkTouchdown(drone) # Check for touchdown
        self.safe_touchdown = self.checkSafeTouchdown(drone) # Check for safe touchdwon
        self.updateControllerFitness(drone,end) # Update the total fitness and energy value

    def orderObstacles(self, pos_drone):
        # Update and order the distancece of the obstacles

        # First update obstacle_distances
        for i in range(len(self.obstacles)):
            self.obstacle_distances[i] = self.obstacles[i].findDistance(pos_drone)
        self.ordered_obstacles = list(np.argsort(self.obstacle_distances)) # Find the sorted of the distances, and reverse

    def findLaserDistances(self, pos_drone):
        # Find the distance of all laser

        right_wall_flag = False
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

            # Check collision with walls
            # right wall (+ x)
            z_wall_right = laser_m*self.x_wall + laser_b
            # check if in the same direction as the laser
            relative_position_wall = np.array([[self.x_wall],[z_wall_right]])-pos_drone
            if laser_vector.dot(relative_position_wall.flatten()) > 0:
                    # set flag to true to not have to check the left wall as well
                    # If contact is above the ground
                    right_wall_flag = True
                    if z_wall_right > 0.:
                        distance_right_wall = np.sqrt((self.x_wall-pos_drone[0][0])**2 + (z_wall_right-pos_drone[1][0])**2)
                        distance = min(distance,distance_right_wall)

            if not right_wall_flag:
                z_wall_left = -laser_m*self.x_wall + laser_b
                if z_wall_left > 0.:
                    distance_left_wall = np.sqrt((-self.x_wall-pos_drone[0][0])**2 + (z_wall_left-pos_drone[1][0])**2)
                    distance = min(distance,distance_left_wall)

            right_wall_flag = False
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
        # Check if drone is on the floor

        if drone.pos[1][0]-(drone.height/2) < 0:
            return True
        return False

    def checkSafeTouchdown(self, drone):
        # Check if drone is on the floor and the safe landing conditions are satisfied

        vel_cond = drone.total_vel <= self.safe_vel # Lower than landing velocity (norm velocity)
        angle_cond = drone.theta_pos <= self.safe_angle or 2*np.pi-drone.theta_pos <= self.safe_angle # Lower than landing angle
        if (drone.pos[1][0]-(drone.height/2) < 0) and vel_cond and angle_cond:
            return True
        return False

    def updateControllerFitness(self, drone, end):
        if self.collision: # if there is a collision with an obstacle
            self.fitness -= 1*2000.
            # self.fitness -= 500 * drone.total_vel
        if end: # if there is no landing by the end of the run
            self.fitness -= 1*1000

        if self.touchdown and not self.safe_touchdown: # If touchdown in unsafe manner
            # self.fitness -= 400

            if drone.theta_pos > np.pi:
                angle_error = (2*np.pi-drone.theta_pos) - self.safe_angle
            else:
                angle_error = drone.theta_pos - self.safe_angle

            theta_vel_error = abs(drone.theta_vel)

            self.fitness -= 5*2 * (abs(self.safe_vel - drone.total_vel) + (50*2/np.pi)*abs(angle_error)+0*theta_vel_error)
        # self.fitness -= (drone.dt/120)*(drone.input_L + drone.input_R)
        self.fitness -= (drone.dt/2)*drone.lasers
        self.energy += drone.lasers*drone.dt

class Obstacle(object):

    def __init__(self,xposition,zposition,radius):
        # Intialise Obstacle object
        self.pos = np.array([[xposition],[zposition]]) # Obstacle Location
        self.radius = radius # Obstacle Radius
        self.shape = Point(xposition,zposition).buffer(radius) # Generate shapely obstacle shape
        self.xcoords, self.zcoords = self.shape.exterior.xy # create the x and y coords for plotting

    def findDistance(self, pos_drone):
        # find distance between obstacle and drone
        return np.linalg.norm(self.pos - pos_drone)

class Course(object):

    def __init__(self):
        # Initialise Course object
        self.obstacles = []

    def default(self):
        # Default course of alternating obstacles
        self.obstacles = []
        total_obstacles = 4
        for i in range(total_obstacles):
            x = 0 - 4*(((total_obstacles-1)/2)-i)
            z = 5
            r = 0.5
            self.obstacles.append(Obstacle(x,z,r))
        self.obstacles.append(Obstacle(0,2,0.5))
        return self.obstacles

    def moreComplicated(self):
        # More complicated version of default course
        self.obstacles = []
        r1 = random.uniform(-1,1)
        r2 = random.uniform(-1,1)
        r3 = random.uniform(-1,1)
        total_obstacles = 4
        for i in range(total_obstacles):
            x = 4*r1 - 2*(((total_obstacles-1)/2)-i)
            z = 5
            r = 0.5
            self.obstacles.append(Obstacle(x,z,r))

        for i in range(total_obstacles):
            x = 4*r2 - 2*(((total_obstacles-1)/2)-i)
            z = 10
            r = 0.5
            self.obstacles.append(Obstacle(x,z,r))

        for i in range(total_obstacles):
            x = 4*r3 - 2*(((total_obstacles-1)/2)-i)
            z = 15
            r = 0.5
            self.obstacles.append(Obstacle(x,z,r))

        self.obstacles.append(Obstacle(0,2,0.5))
        return self.obstacles

    def emptyCourse(self):
        # Empty course used for stabilising controller check
        self.obstacles = []
        self.obstacles.append(Obstacle(40,40,0.5))

        return self.obstacles

    def avoidCourse(self):
        # Course with platforms

        self.obstacles = []

        total_obstacles = 8
        for i in range(total_obstacles):
            x = -6 - 1.1*(((total_obstacles-1)/2)-i)
            z = 15
            r = 0.5
            self.obstacles.append(Obstacle(x,z,r))

        for i in range(total_obstacles):
            x = 6 - 1.1*(((total_obstacles-1)/2)-i)
            z = 10
            r = 0.5
            self.obstacles.append(Obstacle(x,z,r))

        for i in range(total_obstacles):
            x = -3 - 1.1*(((total_obstacles-1)/2)-i)
            z = 5
            r = 0.5
            self.obstacles.append(Obstacle(x,z,r))

        self.obstacles.append(Obstacle(3,2,0.5))
        self.obstacles.append(Obstacle(-4,10,0.5))
        return self.obstacles

    def avoidCourse2(self):
        # Course with multiple gaps

        self.obstacles = []

        total_obstacles =2
        layers = 3
        for j in range(layers):
            modifier = j*1
            for i in range(total_obstacles):
                x = -6+modifier - 1.1*(((total_obstacles-1)/2)-i)
                z = 5*(j+1)
                r = 0.5
                self.obstacles.append(Obstacle(x,z,r))
            for i in range(total_obstacles):
                x = -2+modifier - 1.1*(((total_obstacles-1)/2)-i)
                z = 5*(j+1)
                r = 0.5
                self.obstacles.append(Obstacle(x,z,r))
            for i in range(total_obstacles):
                x = 2+modifier - 1.1*(((total_obstacles-1)/2)-i)
                z = 5*(j+1)
                r = 0.5
                self.obstacles.append(Obstacle(x,z,r))
            for i in range(total_obstacles):
                x = 6+modifier - 1.1*(((total_obstacles-1)/2)-i)
                z = 5*(j+1)
                r = 0.5
                self.obstacles.append(Obstacle(x,z,r))
        return self.obstacles

    def popcornCourse(self,seed):
        # Generate random course with input seed

        self.obstacles = []
        random.seed(seed)
        obstacle_locations = []
        n_obstacles = 20
        min_distance = 3
        while len(self.obstacles) < n_obstacles:
            fail_flag = False
            temp_x = random.uniform(-9.5,9.5)
            temp_y = random.uniform(5.0,24.0)

            for i in range(len(obstacle_locations)):
                distance = np.sqrt((obstacle_locations[i][0]-temp_x)**2+(obstacle_locations[i][1]-temp_y)**2)
                if distance < min_distance:
                    fail_flag = True
                    break

            if not fail_flag:
                obstacle_locations.append((temp_x,temp_y))
                self.obstacles.append(Obstacle(temp_x,temp_y,0.5))
        return self.obstacles
