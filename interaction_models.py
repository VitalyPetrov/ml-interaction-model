#------------------------------------------------------------------------------
# Store the spherically symmetrical interaction models calculators  
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def morse_model(distance, equil_distance=1.0, well_depth=1.0, width=1.0):
    """
    Returns the interaction energy value between two atoms in Morse model
    @param distance: distance between two interacting atoms
    @param equil_distance: equilibrium distance between two atoms (where energy = well_depth)
    @param well_depth: interaction energy value on the equilibrium separation distance
    @param width: ontrols the 'width' of the potential (the smaller width is, the larger the well)
    """
    return well_depth * ( np.exp(-2 * width * (distance - equil_distance)) - 2 * np.exp(-width * (distance - equil_distance)) )

def many_body_morse(distances, equil_distance, well_depth, width):
    """
    Returns the total potential energy value as a sum of Morse pair potentials (==model)
    @param distances: list of pair-wise distances between interacting atoms (dtype = [ [] ])
    @param equil_distance: equilibrium distance between two atoms (where energy = well_depth)
    @param well_depth: interaction energy value on the equilibrium separation distance
    @param width: ontrols the 'width' of the potential (the smaller width is, the larger the well)
    """
    return [sum([morse_model(distance, equil_distance, well_depth, width) for distance in distances_list]) / 2 for distances_list in distances]

if __name__ == "__main__":
    distancies = np.linspace(0.2, 15.0, 100)
    plt.plot(distancies, morse_model(distancies), lw=3.5, marker='o', mec='black')
    plt.show()
