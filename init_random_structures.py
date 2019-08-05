import os
import numpy as np

from ase import Atoms
from ase.calculators.espresso import Espresso
from ase.calculators.calculator import CalculationFailed
from ase.io import Trajectory
from shutil import copyfile, rmtree

def espresso_calculator(element):
    """
    Returns the ESPRESSO calculator object
    """
    pseudopotentials = {'Ar': 'ar_pbe.UPF', 'Cu': 'cu_pbe.UPF', 'Li': 'li_pbe.UPF', 'N': 'n_pbe.UPF'}

    espresso_settings = {
                    'pseudo_dir': './',
                    'input_dft': 'RVV10',
                    'prefix': f'{element}',
                    'electron_maxstep': 100000,
                    'tstress': False,
                    'tprnfor': False,
                    'verbosity': 'low',
                    'occupations': 'smearing',
                    'degauss': 0.05,
                    'smearing': 'marzari-vanderbilt',
                    'ecutwfc': 100
                    }
    espresso_calculator = Espresso(restart = None, pseudopotentials = pseudopotentials,
                                    input_data = espresso_settings)

    return espresso_calculator

def make_simulation_dirs(num_dirs, prefix='simulation', pp_name='pp.data', override=False):
    """ 
    Creates {num_dirs} of directories storing the simulation i/o data 
    @param prefix : naming pattern for directories
    @param pp_name : filename of the pseudopotential to use in simulation
    @param override : specify do you want to override already existing directories
    """
    for idx in range(train_size):
        dir_name = f'{prefix}_{idx}'
        try:
            # create the storing directory
            os.mkdir(dir_name)
            # and copy pseudopotential and rVV10 kernel data files into it
            copyfile(f'{os.getcwd()}/pseudo/{pp_name}', f'{os.getcwd()}/{dir_name}/{pp_name}')
            copyfile(f'{os.getcwd()}/pseudo/rVV10_kernel_table', f'{os.getcwd()}/{dir_name}/rVV10_kernel_table')
        except FileExistsError:
            # if this directory is already exist we can both do nothing or remove it
            # and create the new one depending on {override} parameter
            if override:
                # remove the existing dir
                rmtree(dir_name)
                # and create it again
                os.mkdir(dir_name)
                copyfile(f'{os.getcwd()}/pseudo/{pp_name}', f'{os.getcwd()}/{dir_name}/{pp_name}')
                copyfile(f'{os.getcwd()}/pseudo/rVV10_kernel_table', f'{os.getcwd()}/{dir_name}/rVV10_kernel_table')
                
def create_random_trimer(element):
    """ 
    Creates three randomly distributed atoms in the simulation cell
    @param element: chemical element symbol
    """
    simulation_cell = [ [24., 0., 0.], [0., 24., 0.], [0., 0., 24.] ]

    # creates list of random coordinates for all 3 atoms
    # based on "standard normal” distribution
    random_positions = 6. * np.random.randn(3, 3) + 16.0
    # and finally the atoms object
    atoms = Atoms(f'{element}', calculator=espresso_calculator(element), pbc=False,
                positions=random_positions, cell=simulation_cell)
    return atoms

def get_rotation_matrix(rotation_axis, angle):
    """
    Returns the rotation matrix for selected axis on given angle

    See https://en.wikipedia.org/wiki/Rotation_matrix for more details
    """
    rotation_matrix = np.zeros((3, 3))

    rotation_matrix[0] = [
        np.cos(angle) + (1 - np.cos(angle)) * rotation_axis[0] ** 2,
        (1 - np.cos(angle)) * rotation_axis[0] * rotation_axis[1] - np.sin(angle) * rotation_axis[2],
        (1 - np.cos(angle)) * rotation_axis[0] * rotation_axis[1] + np.sin(angle) * rotation_axis[1]
    ]

    rotation_matrix[1] = [
        (1 - np.cos(angle)) * rotation_axis[0] * rotation_axis[1] + np.sin(angle) * rotation_axis[2],
        np.cos(angle) + (1 - np.cos(angle)) * rotation_axis[1] ** 2,
        (1 - np.cos(angle)) * rotation_axis[1] * rotation_axis[2] - np.sin(angle) * rotation_axis[0]
    ]

    rotation_matrix[2] = [
        (1 - np.cos(angle)) * rotation_axis[2] * rotation_axis[0] - np.sin(angle) * rotation_axis[1],
        (1 - np.cos(angle)) * rotation_axis[2] * rotation_axis[1] + np.sin(angle) * rotation_axis[0],
        np.cos(angle) + (1 - np.cos(angle)) * rotation_axis[2] ** 2
    ]

    return np.array(rotation_matrix)

def create_radial_distributed_trimer(element, distance, angle):
    """
    Creates one atom on cell`s center and two atoms distributed radially 
    on a sphere with {distance} radii
    @param distance: distance between the 1st and 2nd atoms
    @param angle: polar angle between the 1st(or, identically, 2nd) atom and the 3rd one
    """
    simulation_cell = [ [24., 0., 0.], [0., 24., 0.], [0., 0., 24.] ]

    random_positions = [
        3 * [12.], 
        [12., 12., 12. + distance],
        np.array([12., 12., 12.]) + get_rotation_matrix([0., 0., 1.], angle) @ [4., 4., 0.]
    ]
    # and finally the atoms object
    atoms = Atoms(f'{element}', calculator=espresso_calculator(element), pbc=False,
                positions=random_positions, cell=simulation_cell)
    return atoms

def generate_input_scripts(atoms):
    """
    Performs creation of espresso.pwi (input) scripts for Quantum Espresso (QE) code
    Note that this script files are created on the corresponding folder
    """
    try:
        atoms.get_potential_energy()
    except CalculationFailed:
        pass
    # This stuff is needed cause we just want to generate input script
    # not to provide simulation within out python script


if __name__ == "__main__":
    # User-defined params
    min_dist, max_dist = 3, 11 # limits for separation distance between atoms
    train_size = 100 # training set size
    element = '3Cu' # which atoms to simulate
    pp_name = 'cu_pbe.UPF' # filename of the pseudopotential in pseudo/ folder
    configurations = Trajectory('configurations.traj', 'w') # container for all the training configurations
    #
    make_simulation_dirs(num_dirs=train_size, prefix=element, pp_name=pp_name, override=True)
    print('Simulation directories generation: DONE')
    distances = np.linspace(min_dist, max_dist, train_size)

    for idx in range(train_size):
        # change to the particular simulation directory
        os.chdir(f'{element}_{idx}')

        # initialize atoms object
        angle = 4 * np.pi * (np.random.rand() - 0.5)
        atoms = create_radial_distributed_trimer(element, distances[idx], angle)
        generate_input_scripts(atoms)
        configurations.write(atoms)
        #
        print(f'[{idx}/{train_size}]: simulation preprocessing')
        # return to the main dir catalog
        os.chdir('../')