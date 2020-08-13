from render import GeneticPlot
import matplotlib.pyplot as plt
import os

## Plotting for one algorithm run
# plotter = GeneticPlot('/home/benjamin/git/BioInspiredDrone/data/data_2020-08-12_20:15')
# # plotter.plotHistory()
# plotter.plotParetoFront()



# Comparing Algorithm runs
plotters = []
folder_names = os.listdir('important_data')
print(folder_names)
folder_names.sort()
print(folder_names)
for name in folder_names:
    plotters.append(GeneticPlot('/home/benjamin/git/BioInspiredDrone/important_data/'+name))

# Std Deviation
fig_std = plt.figure('Std Deviations')
ax_std = fig_std.add_subplot(1,1,1)
for plotter in plotters:
    plotter.addStdDev(ax_std)
plt.legend()
ax_std.set_xlabel('Generation [-]')
ax_std.set_ylabel('Standard Deviation of Fitness [-]')
# plt.show()

# Max Fitness
fig_max = plt.figure('Max Fitness')
ax_max = fig_max.add_subplot(1,1,1)
for plotter in plotters:
    plotter.addMaxFitnesses(ax_max)
plt.legend()
ax_max.set_xlabel('Generation [-]')
ax_max.set_ylabel('Max Fitness [-]')


plt.show()
