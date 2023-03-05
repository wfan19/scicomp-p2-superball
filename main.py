#!/usr/bin/env python3

import sys
import pickle
import argparse

import numpy as np

from ColloidSim import ColloidSim, ColloidSimParams
from ColloidViz import ColloidViz

try:
    from pyinstrument import Profiler
except:
    print("No profiler found. Not running profiler.")

REAL_TIME = False
PRINT_DEBUG = False
BROWNIAN = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a 3D colloid simulation from a premade list of scenarios")
    parser.add_argument("scenario", default="two_ball")
    args = parser.parse_args()

    if args.scenario == "one_ball":
        # One ball: tunneling/rollback test
        params = ColloidSimParams(
            n_particles = 1,
            box_dims = np.array([1, 1, 1]),
            posns_0 = np.array([[0.49, 0, 0]]).T,
            vels_0= np.array([[1000, 0, 0]]).T,

            default_r = 0.1, # Currently still just support single radius

            # Step and dt to get "real-time" simulations
            # Note that you don't have to follow this; if you crank up the step count or lower dt it'll just be slower than real time
            length=10,  # The simulator runs at 60fps, so this is enough frames for 2mins
            dt=0.01,             # Simulator runs at 60fps
            print_debug=PRINT_DEBUG
        )
    elif args.scenario == "two_ball":
        # Set up basic 2-particle test case: particles going towards each other at the same speed
        params = ColloidSimParams(
            n_particles = 2,
            box_dims = np.array([1, 1, 1]),
            posns_0 = np.array([[0.25, 0, 0], [-0.25, 0, 0]]).T,
            vels_0= np.array([[0.25, 0, 0], [0.5, 0, 0]]).T,

            default_r = 0.1, # Currently still just support single radius

            # Step and dt to get "real-time" simulations
            # Note that you don't have to follow this; if you crank up the step count or lower dt it'll just be slower than real time
            length = 60,  # The simulator runs at 60fps, so this is enough frames for 2mins
            dt=1/60,             # Simulator runs at 60fps
            print_debug=PRINT_DEBUG
        )
    elif args.scenario == "two_ball_chaser":
        # Set up basic 2-particle test case: particles going towards each other at the same speed
        params = ColloidSimParams(
            n_particles = 2,
            box_dims = np.array([1, 1, 1]),
            posns_0 = np.array([[0.25, 0, 0], [-0.25, 0, 0]]).T,
            vels_0= np.array([[0.1, 0, 0], [0.5, 0, 0]]).T,

            default_r = 0.1, # Currently still just support single radius

            # Step and dt to get "real-time" simulations
            # Note that you don't have to follow this; if you crank up the step count or lower dt it'll just be slower than real time
            length = 60,  # The simulator runs at 60fps, so this is enough frames for 2mins
            dt=1/60,             # Simulator runs at 60fps
            print_debug=PRINT_DEBUG
        )
    elif args.scenario == "four_ball":
        # Set up basic 2-particle test case: particles going towards each other at the same speed
        params = ColloidSimParams(
            n_particles = 4,
            box_dims = np.array([1, 1, 1]),

            default_r = 0.25, # Currently still just support single radius

            # Step and dt to get "real-time" simulations
            # Note that you don't have to follow this; if you crank up the step count or lower dt it'll just be slower than real time
            length = 60,  # The simulator runs at 60fps, so this is enough frames for 2mins
            dt=1/60,             # Simulator runs at 60fps
            print_debug=PRINT_DEBUG
        )
    elif args.scenario == "two_ball_brownian":
        # Set up basic 2-particle test case: particles going towards each other at the same speed
        params = ColloidSimParams(
            n_particles = 2,
            box_dims = np.array([1, 1, 1]),
            posns_0 = np.array([[0.25, 0, 0], [-0.25, 0, 0]]).T,
            vels_0= np.array([[0.25, 0, 0], [0.5, 0, 0]]).T,

            default_r = 0.1, # Currently still just support single radius

            brownian=True,
            st_dev=1,

            # Step and dt to get "real-time" simulations
            # Note that you don't have to follow this; if you crank up the step count or lower dt it'll just be slower than real time
            length = 30,  # The simulator runs at 60fps, so this is enough frames for 2mins
            dt=1/60,             # Simulator runs at 60fps
            print_debug=PRINT_DEBUG
        )
    
    elif args.scenario == "100_ball":
        # 10 balls in a small box case.
        params = ColloidSimParams(
            n_particles = 500,
            box_dims = np.array([1, 1, 1]),

            default_r = 0.05, # Currently still just support single radius
            # brownian=True,
            st_dev=0.1,

            # Step and dt to get "real-time" simulations
            # Note that you don't have to follow this; if you crank up the step count or lower dt it'll just be slower than real time
            length = 60,  # The simulator runs at 60fps, so this is enough frames for 2mins
            dt=1/60,             # Simulator runs at 60fps
            print_debug=PRINT_DEBUG
        )
    elif args.scenario == "1000_ball":
        # 10 balls in a small box case.
        params = ColloidSimParams(
            n_particles = 1000,
            box_dims = np.array([1, 1, 1]),

            default_r = 0.05, # Currently still just support single radius
            # brownian=True,
            st_dev=0.1,

            # Step and dt to get "real-time" simulations
            # Note that you don't have to follow this; if you crank up the step count or lower dt it'll just be slower than real time
            length = 60,  # The simulator runs at 60fps, so this is enough frames for 2mins
            dt=1/60,             # Simulator runs at 60fps
            print_debug=PRINT_DEBUG
        )
    else:
        print("Invalid scenario name. Exiting.")
        quit()

    ## Create simulator
    sim = ColloidSim(params)

    ## Create visualizer
    viz = ColloidViz(params, camera_posn=np.array([0.5, 2.5, 1]), control_camera=False)

    if "pyinstrument" in sys.modules:
        # Run profiler if available (for development purposes only)
        profiler = Profiler()
        profiler.start()

    ## Run simulation
    if REAL_TIME:
        for i, t in enumerate(sim.timesteps):

            # TODO: This is copy-pasted from sim.simulate - is there a better way to code-reuse here?
            # If first timestep, initialize the positions and velocities from the given conditions
            if i == 0: # TODO: Also check if initial vels are zero! If initial vels are initialized then don't touch it, just skip
                sim.posns[i, :, :] = sim.params.posns_0
                if not np.any(sim.vels[i, :, :] != 0):
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

    with open(f"{args.scenario}.pkl", "wb") as file:
        pickle.dump(sim, file)