import json
import numpy as np
from hyperevol.tools import particle_swarm as pso
from rosenbrock_scoring import ensemble_rosenbrock


pso_settings = {
    "iterations": 500,
    "population_size": 50,
    "nr_informants": int(np.ceil(0.1 * 10)),
    "output_dir": '/home/laurits/tmp/PSO_test',
    "seed": 1
}


HYPERPARAMETER_INFO = {
    'x': {'max': 500., 'min':-500., 'int': 0, 'exp': 0, 'log': 0, 'power': 0},
    'y': {'max': 500., 'min':-500., 'int': 0, 'exp': 0, 'log': 0, 'power': 0}
}



swarm = pso.ParticleSwarm(ensemble_rosenbrock, HYPERPARAMETER_INFO, **pso_settings)
pso_best_parameters, pso_best_fitness = swarm.particleSwarmOptimization()
print(pso_best_parameters, pso_best_fitness)

