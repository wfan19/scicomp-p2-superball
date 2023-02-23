#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

import streamlit as st

@dataclass
class ColloidSimParams:
    # Simulation timestep and particle counts
    n_steps:        int = 10
    n_particles:    int = 1

    posns_0:        np.ndarray = np.zeros((3, n_particles))
    vels_0:         np.ndarray = np.zeros((3, n_particles))

    # Simulation parameters
    dt:             float = 0.01

    # Dimensions [m]
    box_x:          int = 1
    box_y:          int = 1
    box_z:          int = 1

    particle_r:     float = 0.025 #TODO: Support indiivdual particle sizes

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
        for i, t in enumerate(self.timesteps):

            # If first timestep, initialize the positions and velocities from the given conditions
            if i == 0:
                self.posns[i, :, :] = self.params.posns_0
                self.vels[i, :, :] = self.params.vels_0
                continue

            # 1. Naively propagate forward last velocity
            self.vels[i, :, :] = self.vels[i-1, :, :]

            # 2. Now check for collisions: which velocities do we update?
            # Purely narrow-phase collision checking: Check every possible particle pair
            # TODO: Seperate out broad-phase from narrow phase by using KD or Quad trees.
            for i_particle, particle in enumerate(self.posns[i-1, :, :].T):
                # Get distance between current particle and all other particles
                particle_as_col = np.atleast_2d(particle).T 
                distances = np.linalg.norm(particle_as_col - self.posns[i-1, :, :], axis=0)

                # Check if distances between current particle and any other ones are within the collision threshold
                colliding = (distances <= self.params.particle_r * 2) # TODO: After implementing individual particle sizes, reference from that list
                colliding[i_particle] = False

                if np.any(colliding):
                    i_collision_particle = np.argmin(distances)
                    print(f"Collision between {i_particle} and {i_collision_particle} happened at time {t}")
                    # TODO: Collision handling!

            # 3. Now that we have updated all velocities accordingly, let's integrate the velocities to find the new positions
            # TODO: For entities where a collision after the next step is anticipated, we need to do special things to prevent ghosting
            # r_next = r_current + vel * dt
            self.posns[i, :, :] = self.posns[i-1, :, :] + self.vels[i, :, :] * self.params.dt

if __name__ == "__main__":

    # Set up basic 2-particle test case: particles going towards each other at the same speed
    params = ColloidSimParams(
        n_particles = 2,
        posns_0 = np.array([[0.05, 0, 0], [-0.05, 0, 0]]).T,
        vels_0 = np.array([[-1, 0, 0], [1, 0, 0]]).T
    )
    sim = ColloidSim(params)

    sim.simulate()

    st.write("Simulation Results")
    t = st.slider("Time", 0, params.n_steps - 1, 0)
    fig = go.Figure(go.Scatter3d(
        x=sim.posns[t, 0, :],
        y=sim.posns[t, 1, :],
        z=sim.posns[t, 2, :],
        mode="markers",
        marker={"color": px.colors.qualitative.Dark24, "size": 5}
    ))

    fig.update_layout(
        scene_aspectmode="cube",
        height=800,
        width=800,
        scene={
            "xaxis": {"range": [-params.box_x/2, params.box_x/2]},
            "yaxis": {"range": [-params.box_y/2, params.box_y/2]},
            "zaxis": {"range": [-params.box_z/2, params.box_z/2]},
        },
        uirevision="hi"
    )

    st.plotly_chart(fig)
