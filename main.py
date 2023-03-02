#!/usr/bin/env python3

import sys

import numpy as np

from ColloidSim import ColloidSim, ColloidSimParams
from ColloidViz import ColloidViz

try:
    from pyinstrument import Profiler
except:
    print("No profiler found. Not running profiler.")

SCENARIO = "ten_ball"
REAL_TIME = True
PRINT_DEBUG = True

if __name__ == "__main__":

    if SCENARIO == "two_ball":
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
            dt=1/60,             # Simulator runs at 60fps
            print_debug=PRINT_DEBUG
        )
    
    elif SCENARIO == "ten_ball":
        # 10 balls in a small box case.
        params = ColloidSimParams(
            n_particles = 30,
            box_dims = np.array([1, 1, 1]),

            default_r = 0.1, # Currently still just support single radius

            # Step and dt to get "real-time" simulations
            # Note that you don't have to follow this; if you crank up the step count or lower dt it'll just be slower than real time
            n_steps = int(60*60*0.5),  # The simulator runs at 60fps, so this is enough frames for 2mins
            dt=1/60,             # Simulator runs at 60fps
            print_debug=PRINT_DEBUG
        )
    else:
        print("Invalid scenario name. Exiting.")

    ## Create simulator
    sim = ColloidSim(params)

    ## Create visualizer
    viz = ColloidViz(params, camera_posn=np.array([0.5, 2.5, 1]), control_camera=False)

    if "pyinstrument" in sys.modules:
        profiler = Profiler()
        profiler.start()

    if REAL_TIME:
        for i, t in enumerate(sim.timesteps):

            # TODO: This is copy-pasted from sim.simulate - is there a better way to code-reuse here?
            # If first timestep, initialize the positions and velocities from the given conditions
            if i == 0:
                sim.posns[i, :, :] = sim.params.posns_0
                sim.vels[i, :, :] = sim.params.vels_0
                continue
            
            last_posns = sim.posns[i-1, :, :]
            last_vels = sim.vels[i-1, :, :]
            sim.posns[i, :, :], sim.vels[i, :, :] = sim.step(t, last_posns, last_vels)

            viz.update(sim.posns[i, :, :])
        # Actually, we'll just make it possible for ColloidSim to own a ColloidViz. Then if we want to run it in real time we'll just give it a copy, and if not we'll just not give it a copy.
    else:
        sim.simulate()

        # Feel free to move the camera around
        # control_camera enables fps-style control of the camera, but BEWARE! It's quite broken and will def trip you up :)
        for position_i in sim.posns:
            viz.update(position_i)

    if "pyinstrument" in sys.modules:
        profiler.stop()
        profiler.print()