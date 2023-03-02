#!/usr/bin/env python3

import sys

import numpy as np

from ColloidSim import ColloidSim, ColloidSimParams
from ColloidViz import ColloidViz

try:
    from pyinstrument import Profiler
except:
    print("No profiler found. Not running profiler.")

if __name__ == "__main__":
    if "pyinstrument" in sys.modules:
        profiler = Profiler()
        profiler.start()

    scenario = "ten_ball"

    if scenario == "two_ball":
        # Set up basic 2-particle test case: particles going towards each other at the same speed
        params = ColloidSimParams(
            n_particles = 2,
            box_dims = np.array([1, 1, 1]),
            posns_0 = np.array([[0.25, 0, 0], [-0.25, 0, 0]]).T,
            vels_0= np.array([[1, 0, 0], [0.5, 0, 0]]).T,

            default_r = 0.1, # Currently still just support single radius

            # Step and dt to get "real-time" simulations
            # Note that you don't have to follow this; if you crank up the step count or lower dt it'll just be slower than real time
            n_steps = int(60*60*0.5),  # The simulator runs at 60fps, so this is enough frames for 2mins
            dt=1/60             # Simulator runs at 60fps
        )
    
    elif scenario == "ten_ball":
        # 10 balls in a small box case.
        params = ColloidSimParams(
            n_particles = 30,
            box_dims = np.array([1, 1, 1]),

            default_r = 0.1, # Currently still just support single radius

            # Step and dt to get "real-time" simulations
            # Note that you don't have to follow this; if you crank up the step count or lower dt it'll just be slower than real time
            n_steps = int(60*60*0.5),  # The simulator runs at 60fps, so this is enough frames for 2mins
            dt=1/60             # Simulator runs at 60fps
        )
    else:
        print("Invalid scenario name. Exiting.")

    sim = ColloidSim(params)
    sim.simulate()

    if "pyinstrument" in sys.modules:
        profiler.stop()
        profiler.print()

    ## Create the simulator
    viz = ColloidViz(sim)

    # Feel free to move the camera around
    # control_camera enables fps-style control of the camera, but BEWARE! It's quite broken and will def trip you up :)
    viz.visualize(camera_posn= np.array([0.5, 2.5, 1]),control_camera=False)