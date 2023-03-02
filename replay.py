#!/usr/bin/env python3

import argparse
import pickle

import numpy as np

from ColloidSim import ColloidSim
from ColloidViz import ColloidViz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    with open(args.path, "rb") as file:
        sim = pickle.load(file)
        
        viz = ColloidViz(sim.params, camera_posn=np.array([0.5, 2.5, 1]), control_camera=False)
        for position_i in sim.posns:
            viz.update(position_i)