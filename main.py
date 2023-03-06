#!/usr/bin/env python3

import sys
import pickle
import argparse

import numpy as np

from ColloidSim import ColloidSim, ColloidSimParams
from ColloidViz import ColloidViz
from ResultPlotter import plot_results

try:
    from pyinstrument import Profiler
except:
    print("No profiler found. Not running profiler.")

def run_sim(params: ColloidSimParams, arguments):
    ## Create simulator
    sim = ColloidSim(params)

    ## Create visualizer
    viz = ColloidViz(params, camera_posn=np.array([0.5, 2.5, 1]), control_camera=False)

    if "pyinstrument" in sys.modules:
        # Run profiler if available (for development purposes only)
        profiler = Profiler()
        profiler.start()

    ## Run simulation
    if arguments.pure_sim:
        sim.simulate()
        plot_results(sim)

        # Feel free to move the camera around
        # control_camera enables fps-style control of the camera, but BEWARE! It's quite broken and will def trip you up :)
        for position_i in sim.posns:
            viz.update(position_i)
    else:
        f_update_viz = lambda sim, i, t: viz.update(sim.posns[i, :, :])
        sim.simulate(f_update_viz)
        plot_results(sim)

    if "pyinstrument" in sys.modules:
        profiler.stop()
        profiler.print()

    return sim
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a 3D colloid simulation. You may specify from a list of premade scenarios, or define your own parameters")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("-n", "--n_particles", metavar="N", type=int, default=10, help="Specify the number of particles")
    parser.add_argument("-s", "--std_dev", metavar="STD", type=float, default=1., help="Standard deviation of initial particle velocities")
    parser.add_argument("-b", "--box_dims", nargs=3, metavar=("x", "y", "z"), type=float, default=(1, 1, 1), help="Individual dimensions of box size")
    parser.add_argument("-r", "--radius", metavar="RAD", type=float, default=0.05, help="Radius of all particles. (default 0.1)")
    parser.add_argument("-t", "--time", metavar="TIME", type=int, default=30, help="Lenght of total simulation time in seconds, defaults to 30")
    parser.add_argument("-dt", "--timestep", metavar="TIMESTEP", type=float, default=1/60, help="Length of each timestep in seconds. Defaults to 1/60 for 60fps")
    parser.add_argument("-p", "--pure_sim", action="store_true", help="Toggle whether the simulation will run in real time. If true, update the visuals for each simulation timestep. If false, simulate all timesteps first, then plot everything (default True)")
    parser.add_argument("-f", "--filename", default=None, help="Filename for export pickle file")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Toggle verbose debugging mode. Prints each collision event. (default False).")
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
            print_debug=args.verbose
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
            print_debug=args.verbose
        )
    elif args.scenario == "100_ball":
        # 100 balls in a small box case.
        params = ColloidSimParams(
            n_particles = 100,
            box_dims = np.array([1, 1, 1]),

            default_r = 0.05, # Currently still just support single radius
            st_dev=0.1,

            # Step and dt to get "real-time" simulations
            # Note that you don't have to follow this; if you crank up the step count or lower dt it'll just be slower than real time
            length = 60,  # The simulator runs at 60fps, so this is enough frames for 2mins
            dt=1/60,             # Simulator runs at 60fps
            print_debug=args.verbose
        )
    elif args.scenario is None:
        # If a scenario had not been named, use command line args.
        params = ColloidSimParams(
            n_particles = args.n_particles,
            box_dims = args.box_dims,

            default_r = args.radius, # Currently still just support single radius
            st_dev = args.std_dev,

            # Step and dt to get "real-time" simulations
            # Note that you don't have to follow this; if you crank up the step count or lower dt it'll just be slower than real time
            length = args.time,  # The simulator runs at 60fps, so this is enough frames for 2mins
            dt = args.timestep,             # Simulator runs at 60fps
            print_debug=args.verbose
        )
    else:
        print("Invalid scenario name, exiting")
        quit()

    ## RUN THE SIMULATOR!!
    sim = run_sim(params, args)

    filename = ""
    if args.scenario is not None:
        filename = args.scenario
    elif args.filename is not None:
        filename = args.filename
    else:
        quit()

    with open(f"{filename}.pkl", "wb") as file:
        pickle.dump(sim, file)