#!/usr/bin/env python3

import numpy as np

from ColloidSim import ColloidSim, ColloidSimParams
from ColloidViz import ColloidViz


if __name__ == "__main__":
    # Set up basic 2-particle test case: particles going towards each other at the same speed
    params = ColloidSimParams(
        n_particles = 10,
        box_dims = np.array([1, 1, 1]), # For now please set them to be all the same
        default_r = 0.1,
        n_steps = 60*60*2,
        dt=1/60
    )
    sim = ColloidSim(params)
    sim.simulate()

    viz = ColloidViz(sim)
    viz.visualize(camera_posn= np.array([0.5, 2.5, 1]),control_camera=False)