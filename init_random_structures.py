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
                    'pseudo_dir' : './',
                    'input_dft' : 'RVV10',
                    'prefix' : f'{element}',
                    'electron_maxstep' : 100000,
                    'tstress' : False,
                    'tprnfor' : False,
                    'verbosity' : 'low',
                    'occupations' : 'smearing',
                    'degauss' : 0.05,
                    'smearing' : 'marzari-vanderbilt',
                    'ecutwfc' : 100
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
    simulation_cell = [ [16., 0., 0.], [0., 16., 0.], [0., 0., 16.] ]

    # creates list of random coordinates for all 3 atoms
    # based on "standard normal” distribution
    random_positions = 2.0 * np.random.randn(3, 3) + 8.0
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
    train_size = 100 # training set size
    element = '3Cu' # which atoms to simulate
    pp_name = 'cu_pbe.UPF' # filename of the pseudopotential in pseudo/ folder
    configurations = Trajectory('configurations.traj', 'w') # container for all the training configurations
    #

    make_simulation_dirs(num_dirs=train_size, prefix=element, pp_name=pp_name, override=True)
    print('Simulation directories generation: DONE')
    for idx in range(train_size):
        # change to the particular simulation directory
        os.chdir(f'{element}_{idx}')
        atoms = create_random_trimer(element)
        generate_input_scripts(atoms)
        configurations.write(atoms)

        print(f'[{idx}/{train_size}]: simulation preprocessing')
        # return to the main dir catalog
        os.chdir('../')