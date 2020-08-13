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

        self.threshold = threshold
        self.generation_limit = generation_limit
        self.N_pop = N_pop
        self.parents_in_new_gen = 0.25
        self.mutation_variance = mutation_variance

        # Create Folder name for data storage
        date_time = str(datetime.datetime.now())[:-10]
        temp_list = list(date_time)
        temp_list[10] = '_'
        self.folder_name = './data/data_'+''.join(temp_list)
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

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

        self.parameter_ranges = []
        for i in range(len(self.parameter_limits)):
            self.parameter_ranges.append(self.parameter_limits[i][1]-self.parameter_limits[i][0])

        self.population = self.initialisePopulation()
        self.universal_best_fitness = -100000.

        self.preset = Presets()
        self.preset.loadDefault()

        self.safe_vel = 1.
        self.obstacle_courses = obstacle_courses
        self.simulation_time = 30.0

        # Load stabilisers
        self.stabilisers = []
        for i in range(5):
            self.stabilisers.append(self.loadStabiliser(i+1))

    def initialisePopulation(self):
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

        # Find child population
        crossover_population = []
        normalised_fitnesses = self.normaliseFitness()

        for _ in range(int(len(self.population)*(1-self.parents_in_new_gen))):
            parent_1, parent_2 = np.random.choice(len(self.population), 2, replace=False, p=normalised_fitnesses)
            parent_1_proportion = normalised_fitnesses[parent_1]/(normalised_fitnesses[parent_1]+normalised_fitnesses[parent_2])
            assert(parent_1_proportion < 1. and parent_1_proportion > 0.)
            parent_1 = self.population[parent_1]; parent_2 = self.population[parent_2]
            child = []
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

        original_fitnesses = np.array(self.cummu_fitnesses)
        # original_fitnesses = np.array([-10000, -12, -2, -10000])
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
        print("Last Genome: " + str(new_population[-1]))
        print(len(new_population))

        return new_population

    def run(self):
        self.generation_count = 0


        while self.generation_count <= self.generation_limit:
            print("Running Generation " + str(self.generation_count))
            print("----------------------------------------------")

            self.fitnesses = []
            self.cummu_fitnesses = []
            start = time.time()

            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            # Run processes in parallel
            self.fitnesses = pool.map(self.runOneGenome, self.population.copy())
            # for i in range(len(self.population)):
            #     self.fitnesses.append(self.runOneGenome(self.population[i]))
            for i in range(len(self.fitnesses)):
                self.cummu_fitnesses.append(self.fitnesses[i][0])

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
            self.population = self.newGeneration()
            print("Time Taken: " + str(time.time()-start))

        # If loop done but not reached best fitness
        if self.universal_best_fitness < self.threshold:
            print("Reached Maximum ammount of Generations!")
            self.closeAlgorithm()

    def runOneGenome(self, genome):
        drone, environment, stabiliser, preset_copy = self.decodeGenome(genome)
        fitnesses = []
        information = []
        # For each obstacle course

        for c in range(len(self.obstacle_courses)):

            # Reset all object parameters for new run (if not first run)
            environment.resetEnv(self.obstacle_courses[c])
            drone.resetParams(preset_copy.drone_dict)
            environment.update(drone, False)
            drone.recieveLaserDistances(environment.laser_distances)

            collision = False
            total_t = 0
            while not collision and total_t < self.simulation_time:

                delta_z_dot, delta_x_dot = drone.findDeltaVelocities()
                # create list of inputs

                theta_raw = drone.theta_pos
                if theta_raw < np.pi:
                    theta_input = theta_raw
                else:
                    theta_input = theta_raw - 2*np.pi

                inputs_stabilise = [drone.vel[0][0]+delta_x_dot, drone.vel[1][0]+delta_z_dot, theta_input, drone.theta_vel] # inputs are laser lengths + state information
                # action = net.advance(inputs, preset.dt, preset.dt)
                action = stabiliser.activate(inputs_stabilise)

                drone.update(action[0]*drone.input_limit, action[1]*drone.input_limit)

                if abs(drone.pos[0][0]) > 10.:
                    environment.fitness -= 2000
                    break

                environment.update(drone, False)
                drone.recieveLaserDistances(environment.laser_distances)

                total_t += self.preset.dt
                collision = environment.collision or environment.touchdown

            environment.fitness -= 16*drone.pos[1][0]
            fitnesses.append(environment.fitness)
            information.append([environment.fitness, environment.energy, drone.pos[1][0], drone.vel[1][0], drone.theta_pos])
        # Return the worse fitness
        min_fitness = min(fitnesses)
        idx = fitnesses.index(min_fitness)
        return information[idx]

    def loadStabiliser(self, N):
        local_dir = os.path.dirname(__file__)
        folder_paths = ['first_stabiliser', 'second_stabiliser', 'third_aggressive_stabiliser', 'fourth_stabiliser', 'fifth_aggressive_stabiliser']
        with open(folder_paths[N-1]+'/winner', 'rb') as f:
            stabiliser = pickle.load(f)
        config_path_stabilise = os.path.join(local_dir, folder_paths[N-1]+'/config-nn')
        config_stabilise = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path_stabilise)
        return neat.nn.FeedForwardNetwork.create(stabiliser, config_stabilise)

    def writeData(self, population, fitnesses):
        pop_np = np.array(population)
        # for i in range(len(fitnesses)):
        #     fitnesses[i] = [fitnesses[i]]
        fit_np = np.array(fitnesses)

        # obj_np = np.array(objective_values)
        # full_array = np.concatenate((pop_np, fit_np, obj_np), axis = 1)
        full_array = np.concatenate((pop_np, fit_np), axis = 1)
        np.savetxt(self.folder_name+'/generation_'+str(self.generation_count)+'.txt', full_array )

    def writeBestGenome(self,genome):
        gen_np = np.array(genome.copy())
        np.savetxt(self.folder_name+'/best_genome.txt', gen_np)

    def readBestGenome(self):
        return_list = list(np.loadtxt(self.folder_name+'/best_genome.txt'))
        return_list[8] = int(return_list[8]); return_list[9] = int(return_list[9]); return_list[10] = int(return_list[10])
        return return_list

    def closeAlgorithm(self):
        #
        with open(self.folder_name+'/algorithm_pickle', 'wb') as f:
            pickle.dump(self, f)
