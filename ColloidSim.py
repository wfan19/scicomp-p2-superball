
import numpy as np
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

import streamlit as st

print_debug = False

@dataclass
class ColloidSimParams:
    # Simulation timestep and particle counts
    n_steps:        int = 100
    n_particles:    int = 1

    posns_0:        np.ndarray = None
    vels_0:         np.ndarray = None

    # Simulation parameters
    dt:             float = 0.001

    # Dimensions [m]
    box_x:          int = 1
    box_y:          int = 1
    box_z:          int = 1

    particles_r:    float = None
    particles_mass: np.ndarray = None

    default_r:      float = 0.025
    default_mass:   float = 1

    def __post_init__(self):
        if self.posns_0 is None: self.posns_0 = np.zeros((3, self.n_particles))
        if self.vels_0 is None: self.vels_0 = np.zeros((3, self.n_particles))

        if self.particles_r is None: self.particles_r = self.default_r * np.ones(self.n_particles)
        if self.particles_mass is None: self.particles_mass = self.default_mass * np.ones(self.n_particles)

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

            # Preallocate list of contact normals, which we will populate as we detect collisions
            # This way we only calculate once for each pair
            collision_pairs = set()

            last_posns = self.posns[i-1, :, :]
            last_vels = self.vels[i-1, :, :]

            for i_particle, particle in enumerate(last_posns.T):
                # Get distance between current particle and all other particles
                particle_as_col = np.atleast_2d(particle).T 
                distances = np.linalg.norm(particle_as_col - last_posns, axis=0)

                # Check if distances between current particle and any other ones are within the collision threshold
                # TODO: Elementwise compare against a list of expected collision distances for each particle based on particle.particle_r + collision_particle.particle_r
                colliding = (distances <= self.params.particles_r[i_particle] * 2) 
                colliding[i_particle] = False

                if np.any(colliding):
                    # Find the closest particle - 
                    # NOTE: Hopefully choosing the minimum-distance collision will also auto-resolve any multiple-collision scenario
                    min_nonzero_dist = np.min(distances[np.nonzero(distances)])
                    i_collision_particle = int(np.argwhere(distances == min_nonzero_dist))
                    collision_pair = frozenset((i_particle, i_collision_particle))

                    # This collision has been handled already
                    # Note that since set-retrieval is O(1) via hashing, this is faster than just simply setting the velocities twice.
                    if collision_pair in collision_pairs:
                        continue

                    collision_pairs.add(collision_pair)
                    if print_debug: print(f"Collision between {i_particle} and {i_collision_particle} happened at time {t}")
                    
                    ## Implement general linear collision handling:
                    # See https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects

                    # Pull associated quantities with each particle
                    posn_1, posn_2 = last_posns.T[[i_particle, i_collision_particle]]
                    vel_1, vel_2 = last_vels.T[[i_particle, i_collision_particle]]
                    m_1, m_2 = self.params.particles_mass[[i_particle, i_collision_particle]]

                    # Find the contact normals: (x_1 - x_2) or (x_2 - x_1)
                    # By "contact normal" I mean the vector between the center of mass of each object, which is *normal* to the contact surface. 
                    normal_1 = posn_1 - posn_2                                              # Normal vector centered on "particle"
                    normal_1 = normal_1 / np.linalg.norm(normal_1)                          # Normalize the vector to unit length
                    normal_2 = -normal_1                                                    # Normal vector centered on "collision_particle"

                    # Find the normal components of the original velocities of each particle
                    # This is the component that gets "flipped" by the collision as it's on the line of force. The rest of the velocity is untouched.
                    # Thus we will scale and subtract this from the original 
                    # *Note*: norm(v)^2 is the same as dot(v, v)
                    vel_1_normal = np.dot(vel_1 - vel_2, normal_1) * normal_1
                    vel_2_normal = np.dot(vel_2 - vel_1, normal_2) * normal_2

                    vel_1_new = vel_1 - 2 * m_2 / (m_1 + m_2) * vel_1_normal
                    vel_2_new = vel_2 - 2 * m_1 / (m_1 + m_2) * vel_2_normal

                    self.vels[i, :, i_particle] = vel_1_new
                    self.vels[i, :, i_collision_particle] = vel_2_new

            # 3. Now that we have updated all velocities accordingly, let's integrate the velocities to find the new positions
            # TODO: For entities where a collision after the next step is anticipated, we need to do special things to prevent ghosting
            # r_next = r_current + vel * dt
            self.posns[i, :, :] = self.posns[i-1, :, :] + self.vels[i, :, :] * self.params.dt