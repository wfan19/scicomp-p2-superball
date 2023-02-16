#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass

@dataclass
class ColloidSimParams:
    # Simulation timestep and particle counts
    n_steps:        int = 10
    n_particles:    int = 1

    # Simulation parameters
    dt:             float = 0.01

    # Dimensions [m]
    box_x:          int = 1
    box_y:          int = 1
    box_z:          int = 1

    particle_r:     float = 0.1

class ColloidSim:
    sim_params = ColloidSimParams()

    # Positions and Velocities are both (n_particles x n_timesteps)
    # Each column encodes the position/velocity for all particles per timestep.
    posns:          np.ndarray
    vels:           np.ndarray

    def __init__(self, sim_params: ColloidSimParams = ColloidSimParams()):
        self.sim_params = sim_params

        sim_shape = (sim_params.n_particles, sim_params.n_steps)
        self.posns = np.zeros(sim_shape)
        self.vels = np.zeros(sim_shape)
        
    def simulate(self):
        pass

    def visualize(self):
        pass

if __name__ == "__main__":
    sim_params = ColloidSimParams()

    sim = ColloidSim()

    sim.simulate()

    sim.visualize
