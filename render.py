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
        self.ax.set_xlabel("x Position [m]")
        self.ax.set_ylabel("z Position [m]")
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


class DataStream(object):

    def __init__(self, max_thrust, live):
        # Initialise 4 subplots for z position, linear velocities, angular velocity and inputs
        self.graphs = 4
        self.live = live

        if live:
            plt.ion()
        self.fig = plt.figure()

        self.ax_pos_z = self.fig.add_subplot(self.graphs,1,1)
        self.ax_pos_z.set_ylim(0, 40)
        self.ax_pos_z.set_ylabel("z Position [m]")

        self.ax_vel = self.fig.add_subplot(self.graphs,1,2)
        self.ax_vel.set_ylim(-10, 10)
        self.ax_vel.set_ylabel("Velocity [m/s]")

        self.ax_vel_theta = self.fig.add_subplot(self.graphs,1,3)
        self.ax_vel_theta.set_ylim(-5, 5)
        self.ax_vel_theta.set_ylabel("Angular Velocity [rad/s]")

        self.ax_thrust = self.fig.add_subplot(self.graphs,1,4)
        self.ax_thrust.set_ylim(0, max_thrust*1.1)
        self.ax_thrust.set_ylabel("Thrust Input [N]")
        self.ax_thrust.set_xlabel("Time [s]")

        self.pos_z_line, = self.ax_pos_z.plot([], [], 'b-', label='z position')
        self.ax_pos_z.legend()

        self.vel_x_line, = self.ax_vel.plot([], [], 'r-', label='x velocity')
        self.vel_z_line, = self.ax_vel.plot([], [], 'b-', label='z velocity')
        self.ax_vel.legend()

        self.vel_theta_line, = self.ax_vel_theta.plot([], [], 'k-', label='theta velocity')
        self.ax_vel_theta.legend()

        self.input_L_line, = self.ax_thrust.plot([], [], 'r-', label='input_L')
        self.input_R_line, = self.ax_thrust.plot([], [], 'b-', label='input_R')
        self.ax_thrust.legend()

        self.time_data = []
        self.pos_z_data = []
        self.vel_x_data = []
        self.vel_z_data = []
        self.vel_theta_data = []
        self.input_L_data = []
        self.input_R_data = []


    def updateGraphs(self, drone, total_time):
        # Update data lists
        self.time_data.append(total_time)
        self.pos_z_data.append(drone.pos[1][0])
        self.vel_x_data.append(drone.vel[0][0])
        self.vel_z_data.append(drone.vel[1][0])
        self.vel_theta_data.append(drone.theta_vel)
        self.input_L_data.append(drone.input_L)
        self.input_R_data.append(drone.input_R)

        if self.live:
            self.pos_z_line.set_ydata(self.pos_z_data); self.pos_z_line.set_xdata(self.time_data)
            self.vel_x_line.set_ydata(self.vel_x_data); self.vel_x_line.set_xdata(self.time_data)
            self.vel_z_line.set_ydata(self.vel_z_data); self.vel_z_line.set_xdata(self.time_data)
            self.vel_theta_line.set_ydata(self.vel_theta_data); self.vel_theta_line.set_xdata(self.time_data)
            self.input_L_line.set_ydata(self.input_L_data); self.input_L_line.set_xdata(self.time_data)
            self.input_R_line.set_ydata(self.input_R_data); self.input_R_line.set_xdata(self.time_data)
            self.ax_pos_z.set_xlim(0, self.time_data[-1]*1.1)
            self.ax_vel.set_xlim(0, self.time_data[-1]*1.1)
            self.ax_vel_theta.set_xlim(0, self.time_data[-1]*1.1)
            self.ax_thrust.set_xlim(0, self.time_data[-1]*1.1)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def plotEnd(self,pause):
        self.pos_z_line.set_ydata(self.pos_z_data); self.pos_z_line.set_xdata(self.time_data)
        self.vel_x_line.set_ydata(self.vel_x_data); self.vel_x_line.set_xdata(self.time_data)
        self.vel_z_line.set_ydata(self.vel_z_data); self.vel_z_line.set_xdata(self.time_data)
        self.vel_theta_line.set_ydata(self.vel_theta_data); self.vel_theta_line.set_xdata(self.time_data)
        self.input_L_line.set_ydata(self.input_L_data); self.input_L_line.set_xdata(self.time_data)
        self.input_R_line.set_ydata(self.input_R_data); self.input_R_line.set_xdata(self.time_data)
        self.ax_pos_z.set_xlim(0, self.time_data[-1]*1.2)
        self.ax_vel.set_xlim(0, self.time_data[-1]*1.2)
        self.ax_vel_theta.set_xlim(0, self.time_data[-1]*1.2)
        self.ax_thrust.set_xlim(0, self.time_data[-1]*1.2)
        plt.ion()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(pause)
