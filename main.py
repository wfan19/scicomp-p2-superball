#!/usr/bin/env python3

import numpy as np

from ColloidSim import ColloidSim, ColloidSimParams
from ColloidViz import ColloidViz


if __name__ == "__main__":
    # Set up basic 2-particle test case: particles going towards each other at the same speed
    params = ColloidSimParams(
        n_particles = 10,
        box_dims = np.array([1, 1, 1]),

        default_r = 0.1, # Currently still just support single radius

        # Step and dt to get "real-time" simulations
        # Note that you don't have to follow this; if you crank up the step count or lower dt it'll just be slower than real time
        n_steps = 60*60*2,  # The simulator runs at 60fps, so this is enough frames for 2mins
        dt=1/60             # Simulator runs at 60fps
    )
    sim = ColloidSim(params)
    sim.simulate()

    viz = ColloidViz(sim)

    # Feel free to move the camera around
    # control_camera enables fps-style control of the camera, but BEWARE! It's quite broken and will def trip you up :)
    viz.visualize(camera_posn= np.array([0.5, 2.5, 1]),control_camera=False)