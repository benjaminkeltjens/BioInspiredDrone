"""
This file is the renderer class using shapely and matplotlib

Benjamin Keltjens
July 2020
"""
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import numpy as np


class Renderer(object):

    def __init__(self, obstacles, drone, xlims, ylims):
        # Initialise the graph

        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        plt.xlim(xlims[0], xlims[1])
        plt.ylim(ylims[0], ylims[1])
        plt.gca().set_aspect('equal', adjustable='box')
        self.obstacle_lines = self.createObstacles(obstacles)
        self.laser_lines = self.initialiseLasers(drone)
        self.drone_line = self.initialiseDrone(drone)
        self.ground_line, = self.ax.plot([xlims[0], xlims[1]], [0, 0], 'g-')

    def updateGraph(self, drone):
        # Update Drone
        self.drone_line.set_xdata(drone.xcoords)
        self.drone_line.set_ydata(drone.zcoords)
        self.updateLasers(drone)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def createObstacles(self, obstacles):
        # Generate all the obstacle lines and save to list for debugging
        lines = []
        for obstacle in obstacles:
            line, = self.ax.plot(obstacle.xcoords, obstacle.zcoords, 'b-')

            lines.append(line)
        return lines

    def initialiseLasers(self, drone):
        # Generate all the obstacle lines and save to list for debugging and editting
        x0 = drone.pos[0][0]
        z0 = drone.pos[1][0]
        lines = []
        for i in range(drone.lasers):
            angle = drone.laser_list[i]
            distance = drone.laser_distances[i]

            if distance > 1000: # Make sure distance not too large
                distance = 500

            x1 = x0 + distance*np.cos(angle)
            z1 = z0 - distance*np.sin(angle)
            line, = self.ax.plot([x0, x1], [z0, z1], 'r-')
            lines.append(line)

        return lines

    def updateLasers(self, drone):
        # Update lines for lasers
        x0 = drone.pos[0][0]
        z0 = drone.pos[1][0]
        for i in range(drone.lasers):
            angle = drone.laser_list[i]
            distance = drone.laser_distances[i]

            if distance > 1000: # Make sure distance not too large
                distance = 500

            x1 = x0 + distance*np.cos(angle)
            z1 = z0 - distance*np.sin(angle)
            self.laser_lines[i].set_xdata([x0, x1])
            self.laser_lines[i].set_ydata([z0, z1])




    def initialiseDrone(self, drone):
        line, = self.ax.plot(drone.xcoords, drone.zcoords, 'k-')
        return line
