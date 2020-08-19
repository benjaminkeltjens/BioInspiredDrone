from render import GeneticPlot
import matplotlib.pyplot as plt
import os
import numpy as np

# Plot Pareto Front
# Plotting for one algorithm run
plotter = GeneticPlot('./important_data/1.0')
# plotter.plotHistory()
plotter.plotParetoFront()

#-----------------------------------------------------------------------------------

# Comparing Algorithm runs
# Load data from folders
plotters = []
folder_names = os.listdir('important_data')
print(folder_names)
folder_names.sort()
print(folder_names)
for name in folder_names:
    plotters.append(GeneticPlot('./important_data/'+name))

# Std Deviation
fig_std = plt.figure('Std Deviations')
plt.rcParams.update({'font.size': 15})
ax_std = fig_std.add_subplot(1,1,1)
for plotter in plotters:
    plotter.addStdDev(ax_std)
plt.legend()
ax_std.set_xlabel('Generation [-]')
ax_std.set_ylabel('Standard Deviation of Fitness [-]')

# Max Fitness
fig_max = plt.figure('Max Fitness')
plt.rcParams.update({'font.size': 15})
ax_max = fig_max.add_subplot(1,1,1)
for plotter in plotters:
    plotter.addMaxFitnesses(ax_max)
plt.legend()
ax_max.set_xlabel('Generation [-]')
ax_max.set_ylabel('Max Fitness [-]')


plt.show()

#-----------------------------------------------------------------------------------

# Find genome characteristics on max fitness genomes
angle_ranges = []
for i in range(12):
    angle_ranges.append((np.pi/6)+i*(np.pi/6))

parameter_limits = [[0,3], #z_sat
    [0,3], #x_sat
    [1,10], #z_lim
    [1,10], #x_lim
    [0,3], #z_norm
    [0,3], #x_norm
    [0,1], #z_up
    [2,5], #max_vel
    [0,len(angle_ranges)-1], #angle range choice (index)
    [2,20], #number of lasers
    [1,5]] #stabiliser_choice

parameter_ranges = []
for i in range(len(parameter_limits)):
    parameter_ranges.append(parameter_limits[i][1]-parameter_limits[i][0])

best_genomes = []
for name in folder_names:
    genome = list(np.loadtxt('./important_data/'+name+'/best_genome.txt'))
    genome[8] = int(genome[8]); genome[9] = int(genome[9]); genome[10] = int(genome[10])
    best_genomes.append(genome)

labels = ['z_sat', 'x_sat', 'z_lim', 'x_lim', 'z_norm', 'x_norm', 'z_up', 'max_vel', 'angle_range', 'number of lasers', 'stabiliser']

for i in range(len(parameter_limits)):
    print(labels[i])
    values = []
    for j in range(len(best_genomes)):
        print(folder_names[j] + ': ' + str(best_genomes[j][i]))
        if labels[i] == 'angle_range':
            values.append(angle_ranges[best_genomes[j][i]])
        else:
            values.append(best_genomes[j][i])
    print('Average = ' + str(np.mean(values)))
    print('Std Dev = ' + str(np.std(values)))
    print('Normalise Std Dev = ' + str(np.std(values)/parameter_ranges[i]))
    print('--------------------------------------')

#-----------------------------------------------------------------------------------
