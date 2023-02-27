#!/usr/bin/env python3

import numpy as np

from ColloidSim import ColloidSim, ColloidSimParams
from ColloidViz import ColloidViz


if __name__ == "__main__":
    # Set up basic 2-particle test case: particles going towards each other at the same speed
    params = ColloidSimParams(
        n_particles = 2,
        posns_0 = np.array([[0.5, 0.01, 0.005], [-0.5, 0, 0]]).T,
        vels_0 = np.array([[-2, 0, 0], [1, 0, 0]]).T,
        default_r = 0.1,
        n_steps = 1000,
        dt=0.01
    )
    sim = ColloidSim(params)
    sim.simulate()

    viz = ColloidViz(sim)
    viz.visualize()