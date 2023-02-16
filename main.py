#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass

import matplotlib.pyplot as plt

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
    params = ColloidSimParams()

    # Positions and Velocities are both (n_timesteps * n_particles * n_timesteps)
    # Each "layer" (first dimension) corresponds to a single timestep
    # Each column encodes the position/velocity for a single particle per timestep
    # Each row corresponds to the value's x-y-z dimension
    times:          np.ndarray
    posns:          np.ndarray
    vels:           np.ndarray

    def __init__(self, params: ColloidSimParams = ColloidSimParams()):
        self.params = params

        sim_shape = (params.n_steps, 3, params.n_particles)
        self.posns = np.zeros(sim_shape)
        self.vels = np.zeros(sim_shape)
        
        t_start = 0
        t_end = t_start + params.dt * params.n_steps
        self.timesteps = np.linspace(t_start, t_end, params.n_steps)

    def simulate(self):
        # Placeholder constant velocity
        self.vels[:, 0, :] = 1

        for i, t in enumerate(self.timesteps):
            if i == 0:
                # TODO: Initialize initial positions at some point? But also maybe not here.
                continue

            # Naive discrete kinematics
            # r_next = r_current + vel * dt
            self.posns[i, :, :] = self.posns[i-1, :, :] + self.vels[i, :, :] * self.params.dt

    def visualize(self):
        pass

if __name__ == "__main__":
    params = ColloidSimParams()
    sim = ColloidSim(params)

    sim.simulate()
    breakpoint()

    sim.visualize
