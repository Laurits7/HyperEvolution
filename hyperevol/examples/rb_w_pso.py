''' This script runs the Particle swarm optimization (PSO) with a given batch
size for 1000 repeats each optimization consists out of 10k total evaluations.
Call with 'python'

Usage: rb_w_pso.py --output_dir=DIR

Options:
    -o --output_dir=DIR             Directory of the output
'''
import os
import json
import docopt
import numpy as np
from hyperevol.tools import particle_swarm as pso
from rosenbrock_scoring import ensemble_rosenbrock


def read_cfg(path: str) -> dict:
    ''' Reads the .json file for the settings

    Args:
        path : str
            Path to the configuration file

    Returns:
        cfg : dict
            Configuration read from the given path
    '''
    with open(path, 'rt') as cfg_file:
        cfg = json.load(cfg_file)
    return cfg


def save_results(parameters: dict, fitness: float, output_dir: str) -> None:
    ''' Saves the results to two separate files: parameters to
    "optimal_parameters.json" and fitness score to "fitness.json"

    Args:
        parameters : dict
            The best found parameters to be saved
        fitness : float
            The fitness corresponding to the best found parameters
        output_dir : str
            The directory where the output will be written

    Returns:
        None
    '''
    param_out_path = os.path.join(output_dir, 'optimal_parameters.json')
    fitness_out_path = os.path.join(output_dir, 'fitness.json')
    with open(param_out_path, 'wt') as param_file:
        json.dump(parameters, param_file, indent=4)
    with open(fitness_out_path, 'wt') as fitness_file:
        json.dump(fitness, fitness_file, indent=4)
    print(f"Results saved to:\n\t{param_out_path} \n and\n\t{fitness_out_path}")


def main(output_dir: str) -> None:
    ''' Runs the particle swarm optimization to optimize the Rosenbrock function
    and saves the result to a file in the specified folder.
        Since no additional parameters need to be given to the Rosenbrock fn,
    then no additional 'settings=xyz' will be specified for PSO here.
        After optimization, the other logging info (e.g. score evolution) can
    be accessed easily by e.g "swarm.global_bests"

    Args:
        output_dir : str
            The directory where the output will be written

    Returns:
        None
    '''
    os.makedirs(output_dir, exist_ok=True)
    pso_cfg = read_cfg('config/pso_cfg.json')
    hyperparameters = read_cfg('config/rosenbrock_cfg.json')
    swarm = pso.ParticleSwarm(ensemble_rosenbrock, hyperparameters, **pso_cfg)
    pso_best_parameters, pso_best_fitness = swarm.optimize()
    print(f"Found optimal parameters: {pso_best_parameters}")
    print(f"Found optimal value with optimal parameters: {pso_best_fitness}")
    print("--------------------------------------------------------")
    print("Saving results:")
    save_results(pso_best_parameters, pso_best_fitness, output_dir)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        output_dir = arguments['--output_dir']
        main(output_dir)
    except docopt.DocoptExit as e:
        print(e)
