#------------------------------------------------------------------------------
# Performs potential energy surface (PES) model training utilizing
# numerical optimization techniques
#------------------------------------------------------------------------------
import numpy as np
import fnmatch

from interaction_models import many_body_morse
from scipy.optimize import curve_fit
from glob import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ase import Atoms
from ase.io import read

class InteractionModelTraining:
    """ 
    Interatomic potenital training on the results of DFT simulations 
    To initialize, specify:
    @param dir_names: naming pattern for the simulation directories
    @param prefix: prefix name of output file <-> job title on launch script
    """
    def __init__(self, dir_names='pwscf', prefix='espresso'):
        """
        @param dir_names: naming pattern for the simulation directories
        @param prefix: prefix name of output file <-> job title on launch script
        """
        self.dir_names = dir_names
        self.prefix = prefix

        # Reference pairwise distances and energies
        self.distances, self.ref_energies = self.__process_output()
        # Interatomic model coeffs
        self.model_coeffs = None


    def __get_pairwise_distances(self, atoms):
        """
        Returns the list of the pairwise distances between all the atoms
        @param atoms: ASE Atoms object
        """
        return [dist for dist in np.unique(atoms.get_all_distances()) if dist != 0.]

    def __process_output(self):
        """
        Launch the post-processing of output files generated by Quantum Espresso code
        Returns arrays of atoms potential energy and corresponding pairwise distances
        """
        simulation_dirs = glob(f'{self.dir_names}*')
        # read Atoms object from each output file
        configurations = []
        for dir in simulation_dirs:
            try:
                fout = glob(f'{dir}/{self.prefix}.o*')[0]
                configurations.append(read(fout))
            except IndexError:
                print(f'Invalid output file on directory: {dir}. This directory is ignored')

        distances = [self.__get_pairwise_distances(configuration) for configuration in configurations]
        ref_energies = [configuration.get_potential_energy() for configuration in configurations]

        return distances, ref_energies

    def fit(self):
        """
        Performs model parameters fitting based on the data from QE simulations 
        in order to obtain trained model
        Returns the optimal parameters values
        """
        self.model_coeffs, _ = curve_fit(many_body_morse, self.distances, self.ref_energies)
        return self.model_coeffs

    def evaluate_error(self, metric='MSE'):
        """
        Compute the prediction error corresponding to certain metric
        @param metric: can be one of 'MSE'(mean square error, default) or 'MAE' (mean absolute error)
        """
        predicted_energies = many_body_morse(self.distances, *self.model_coeffs)

        if metric == 'MSE':
            return mean_squared_error(self.ref_energies, predicted_energies)
        elif metric == 'MAE':
            return mean_absolute_error(self.ref_energies, predicted_energies)



if __name__ == "__main__":
    model_training = InteractionModelTraining(dir_names='3Cu', prefix='pes_espresso')
    print(f'Coeffs values: {model_training.fit()}')
    print(f'Error: {model_training.evaluate_error()}')