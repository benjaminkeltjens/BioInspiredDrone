"""
This file describes the GeneticAlgorithm class

Benjamin Keltjens
July 2020
"""

import random
import numpy as np
import pickle
import multiprocessing
import os
import time
import datetime
import matplotlib.pyplot as plt

from drone import AvoiderDrone
from environment import Environment, Obstacle, Course
from presets import Presets

import neat

class GeneticAlgorithm(object):

    def __init__(self, threshold, generation_limit, N_pop, mutation_variance, obstacle_courses):
        # Intialise Class for Genetic Algorithm

        self.threshold = threshold # Threshold on fitness value
        self.generation_limit = generation_limit # Number of generations before algorithm ends
        self.N_pop = N_pop # Population Size
        self.parents_in_new_gen = 0.25 # Percentage of parents transfered to the new generation
        self.mutation_variance = mutation_variance # Variance of the normal distribution for mutation

        # Create Folder name for data storage
        date_time = str(datetime.datetime.now())[:-10]
        temp_list = list(date_time)
        temp_list[10] = '_'
        self.folder_name = './data/data_'+''.join(temp_list)
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

        # Find the range of angles for the lasers
        self.angle_ranges = []
        for i in range(12):
            self.angle_ranges.append((np.pi/6)+i*(np.pi/6))

        self.parameter_limits = [[0,3], #z_sat
            [0,3], #x_sat
            [1,10], #z_lim
            [1,10], #x_lim
            [0,3], #z_norm
            [0,3], #x_norm
            [0,1], #z_up
            [2,5], #max_vel
            [0,len(self.angle_ranges)-1], #angle range choice (index)
            [2,20], #number of lasers
            [1,5]] #stabiliser_choice

        # Find the Range of the parameters
        self.parameter_ranges = []
        for i in range(len(self.parameter_limits)):
            self.parameter_ranges.append(self.parameter_limits[i][1]-self.parameter_limits[i][0])

        # Intialise Population
        self.population = self.initialisePopulation()
        self.universal_best_fitness = -100000.

        # Load in the drone presets
        self.preset = Presets()
        self.preset.loadDefault()

        # Set parameters and courses
        self.safe_vel = 1.
        self.obstacle_courses = obstacle_courses
        self.simulation_time = 30.0

        # Load stabiliser controllers
        self.stabilisers = []
        for i in range(5):
            self.stabilisers.append(self.loadStabiliser(i+1))

    def initialisePopulation(self):
        # Intialise the population of genomes randomly

        population = []

        for _ in range(self.N_pop):
            genome = []
            for i in range(len(self.parameter_limits)):
                if i <= 7:
                    genome.append(random.uniform(self.parameter_limits[i][0],self.parameter_limits[i][1]))
                else:
                    genome.append(random.randint(self.parameter_limits[i][0],self.parameter_limits[i][1]))
            population.append(genome)

        return population

    def decodeGenome(self, genome):
        # Given the genome create the genome, environment and stabiliser objects

        # Drone
        preset_copy = self.preset
        preset_copy.drone_dict["laser_range"] = self.angle_ranges[genome[8]]
        preset_copy.drone_dict["lasers"] = genome[9]
        drone = AvoiderDrone(preset_copy.drone_dict, genome[0], genome[1], genome[2], genome[3], genome[4], genome[5], genome[6],
                            self.safe_vel, genome[7])

        # Environment
        # Load with first obstacle course
        environment = Environment(genome[9], self.obstacle_courses[0], preset_copy.max_laser_length, self.safe_vel, preset_copy.safe_angle)

        # Stabiliser
        stabiliser = self.stabilisers[genome[10]-1]
        return drone, environment, stabiliser, preset_copy

    def crossover(self):
        # Crossover genomes in the current population and select genomes to transfer to next genome pool

        # Find child population
        crossover_population = []
        # Normalise the fitnesses
        normalised_fitnesses = self.normaliseFitness()

        for _ in range(int(len(self.population)*(1-self.parents_in_new_gen))):
            # Select Parent based on fitness
            parent_1, parent_2 = np.random.choice(len(self.population), 2, replace=False, p=normalised_fitnesses)
            # Find the relative fitnesses of the two parents
            parent_1_proportion = normalised_fitnesses[parent_1]/(normalised_fitnesses[parent_1]+normalised_fitnesses[parent_2])
            assert(parent_1_proportion < 1. and parent_1_proportion > 0.)
            # Get the genomes of the parents
            parent_1 = self.population[parent_1]; parent_2 = self.population[parent_2]
            child = []
            # Use averages to find the resultant genome of the chile and add to crossover population
            for i in range(len(parent_1)):
                if i <= 7: # all the continuous variables
                    child.append(parent_1[i]*parent_1_proportion+parent_2[i]*(1-parent_1_proportion))
                else: #integer variables
                    child.append(round(parent_1[i]*parent_1_proportion+parent_2[i]*(1-parent_1_proportion)))
            crossover_population.append(child)

        # Add previous generation to new generation
        parents_to_new = np.random.choice(len(self.population), int(len(self.population)*self.parents_in_new_gen)-1, replace=False, p=normalised_fitnesses)
        for idx in parents_to_new:
            crossover_population.append(self.population[idx])

        return crossover_population

    def mutation(self, population):
        # Mutate crossover population using normal distribution


        mutated_population = []
        p_range = 0.25 # percentage of the range that the first standard deviation represents
        # For every genome
        for n in range(len(population)):
            genome = population[n]
            # For every genome type
            for i in range(len(genome)):
                if i <= 7: # all the continuous variables
                    new_value = genome[i] + p_range*(self.parameter_ranges[i]/2)*np.random.normal(scale=self.mutation_variance)
                    while (new_value < self.parameter_limits[i][0]) or (new_value > self.parameter_limits[i][1]): # If not in parameter limits redo mutation
                        new_value = genome[i] + p_range*(self.parameter_ranges[i]/2)*np.random.normal(scale=self.mutation_variance)
                    genome[i] = float(new_value)
                else: # integer variables
                    new_value = round(genome[i] + p_range*(self.parameter_ranges[i]/2)*np.random.normal(scale=self.mutation_variance))
                    while (new_value < self.parameter_limits[i][0]) or (new_value > self.parameter_limits[i][1]):
                        new_value = round(genome[i] + p_range*(self.parameter_ranges[i]/2)*np.random.normal(scale=self.mutation_variance))
                    genome[i] = int(new_value)

            mutated_population.append(genome)

        return mutated_population

    def normaliseFitness(self):
        # Normalise the fitnesses to rank genomes

        original_fitnesses = np.array(self.cummu_fitnesses)
        original_fitnesses -= np.min(original_fitnesses)

        if sum(original_fitnesses) == 0.:
            # If all the same return list of uniform probability
            return [1/len(original_fitnesses)]*len(original_fitnesses)

        original_fitnesses += 0.0001 # To avoid probability of 0

        normalised_fitnesses = original_fitnesses/np.sum(original_fitnesses)

        if sum(normalised_fitnesses > 0.) == 1: # If there is only 1 one and all other zeros then just mix everything up anyways
            return [1/len(original_fitnesses)]*len(original_fitnesses)

        return list(normalised_fitnesses)

    def newGeneration(self):
        print("\n Producing New Generation")
        print("\n Performing Crossover")
        crossover_population = self.crossover()

        print("\n Performing Mutation")
        new_population = self.mutation(crossover_population)
        new_population.append(self.readBestGenome())
        # Add best genome to population to avoid losing knowledge
        print("Last Genome: " + str(new_population[-1]))
        print(len(new_population))

        return new_population

    def run(self):
        # Run the main algorithm

        self.generation_count = 0


        while self.generation_count <= self.generation_limit:
            print("Running Generation " + str(self.generation_count))
            print("----------------------------------------------")

            self.fitnesses = []
            self.cummu_fitnesses = []
            start = time.time()

            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            # Run simulations in parallel
            self.fitnesses = pool.map(self.runOneGenome, self.population.copy())

            # Seperate the fitness scores from all the fitness information
            for i in range(len(self.fitnesses)):
                self.cummu_fitnesses.append(self.fitnesses[i][0])

            # Write data for generation to file
            self.writeData(self.population.copy(),self.fitnesses.copy())

            # If best genome so far store it!
            if max(self.cummu_fitnesses) > self.universal_best_fitness:
                print("CHANGING BEST RUN")
                self.universal_best_generation = self.generation_count
                self.universal_best_fitness = max(self.cummu_fitnesses)
                self.writeBestGenome(self.population[self.cummu_fitnesses.index(max(self.cummu_fitnesses))])

            print("Best Historical fitness: " + str(self.universal_best_fitness))
            print("Best Historical Genome: " + str(self.readBestGenome()))
            assert(self.runOneGenome(self.readBestGenome())[0] == self.universal_best_fitness)

            print("Best Fitness is: " + str(max(self.cummu_fitnesses)))
            print("Average Fitness is: " + str(sum(self.cummu_fitnesses)/len(self.cummu_fitnesses)))

            if max(self.cummu_fitnesses) >= self.threshold:
                print("Achieved Threshold Fitness Value!")
                self.closeAlgorithm()
                break

            self.generation_count += 1
            # Generate new population
            self.population = self.newGeneration()
            print("Time Taken: " + str(time.time()-start))

        # If loop done but not reached best fitness
        if self.universal_best_fitness < self.threshold:
            print("Reached Maximum ammount of Generations!")
            self.closeAlgorithm()

    def runOneGenome(self, genome):
        # Decode Genome
        drone, environment, stabiliser, preset_copy = self.decodeGenome(genome)
        fitnesses = []
        information = []
        # For each obstacle course

        for c in range(len(self.obstacle_courses)):

            # Reset all object parameters for new run
            environment.resetEnv(self.obstacle_courses[c])
            drone.resetParams(preset_copy.drone_dict)
            environment.update(drone, False)
            drone.recieveLaserDistances(environment.laser_distances)

            collision = False
            total_t = 0
            # Run simulation of course
            while not collision and total_t < self.simulation_time:

                delta_z_dot, delta_x_dot = drone.findDeltaVelocities()
                # create list of inputs

                # Convert theta to positive-negative representation about positive x-axis
                theta_raw = drone.theta_pos
                if theta_raw < np.pi:
                    theta_input = theta_raw
                else:
                    theta_input = theta_raw - 2*np.pi

                inputs_stabilise = [drone.vel[0][0]+delta_x_dot, drone.vel[1][0]+delta_z_dot, theta_input, drone.theta_vel] # inputs are laser lengths + state information

                # Input to the stabiliser network
                action = stabiliser.activate(inputs_stabilise)

                # Simulate drone dynamics and laser calculations
                drone.update(action[0]*drone.input_limit, action[1]*drone.input_limit)

                # Check that drone has not left the course
                if abs(drone.pos[0][0]) > 10.:
                    environment.fitness -= 2000
                    break

                # Update Dynamics, Environment and Fitness
                environment.update(drone, False)
                drone.recieveLaserDistances(environment.laser_distances)

                # Update time and check for any collisions
                total_t += self.preset.dt
                collision = environment.collision or environment.touchdown

            # Penalise z position
            environment.fitness -= 16*drone.pos[1][0]
            fitnesses.append(environment.fitness)
            # Form list of relavent information (fitness, energy, z_pos, z_vel, theta) at the end of the run
            information.append([environment.fitness, environment.energy, drone.pos[1][0], drone.vel[1][0], drone.theta_pos])
        # Return the worse fitness information
        min_fitness = min(fitnesses)
        idx = fitnesses.index(min_fitness)
        return information[idx]

    def loadStabiliser(self, N):
        # Load stabiliser network from folder

        local_dir = os.path.dirname(__file__)
        folder_paths = ['first_stabiliser', 'second_stabiliser', 'third_stabiliser', 'fourth_stabiliser', 'fifth_stabiliser']
        with open(folder_paths[N-1]+'/winner', 'rb') as f:
            stabiliser = pickle.load(f)
        config_path_stabilise = os.path.join(local_dir, folder_paths[N-1]+'/config-nn')
        config_stabilise = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path_stabilise)
        return neat.nn.FeedForwardNetwork.create(stabiliser, config_stabilise)

    def writeData(self, population, fitnesses):
        # Write generation data to file

        # Convert information to numpy arrays
        pop_np = np.array(population)
        fit_np = np.array(fitnesses)

        # Concatenate array and save as txt file
        full_array = np.concatenate((pop_np, fit_np), axis = 1)
        np.savetxt(self.folder_name+'/generation_'+str(self.generation_count)+'.txt', full_array )

    def writeBestGenome(self,genome):
        # Update the file containting the best genome in array format
        gen_np = np.array(genome.copy())
        np.savetxt(self.folder_name+'/best_genome.txt', gen_np)

    def readBestGenome(self):
        # Read genome array from txt file
        return_list = list(np.loadtxt(self.folder_name+'/best_genome.txt'))
        return_list[8] = int(return_list[8]); return_list[9] = int(return_list[9]); return_list[10] = int(return_list[10])
        return return_list

    def closeAlgorithm(self):
        # Pickle this object
        with open(self.folder_name+'/algorithm_pickle', 'wb') as f:
            pickle.dump(self, f)
