#!/usr/bin/env python3

import numpy as np

from ColloidSim import ColloidSim, ColloidSimParams
from ColloidViz import ColloidViz


if __name__ == "__main__":
    # Set up basic 2-particle test case: particles going towards each other at the same speed
    params = ColloidSimParams(
        n_particles = 2,
        posns_0 = np.array([[0.05, 0.01, 0.005], [-0.05, 0, 0]]).T,
        vels_0 = np.array([[-1, 0, 0], [1, 0, 0]]).T
    )
    sim = ColloidSim(params)
    sim.simulate()

    viz = ColloidViz
    viz.visualize(sim)