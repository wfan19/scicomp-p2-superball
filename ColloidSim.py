
import numpy as np
from dataclasses import dataclass, field

@dataclass
class ColloidSimParams:
    # Simulation timestep and particle counts
    n_particles:    int = 1

    posns_0:        np.ndarray = None
    vels_0:         np.ndarray = None

    # Simulation parameters
    length:         float = 1
    dt:             float = 1/60
    n_steps:        int = -1

    # Dimensions [m]
    box_dims:       np.ndarray = np.array([2, 2, 2])

    particles_r:    float = None
    particles_mass: np.ndarray = None

    default_r:      float = 0.025
    default_mass:   float = 1

    st_dev:         float = 0.5
    print_debug:    bool = False

    seed:           int = 1
    rng:            np.random.Generator = None

    def __post_init__(self):
        self.n_steps = int(self.length / self.dt)

        self.rng = np.random.default_rng(1)
        posns_max = np.reshape(self.box_dims, (3, 1)) / 2

        if self.posns_0 is None: 
            self.posns_0 = self.rng.uniform(-posns_max, posns_max, (3, self.n_particles))
        if self.vels_0 is None: 
            self.vels_0 = self.rng.normal(0, self.st_dev, (3, self.n_particles))

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
        self.times = np.linspace(t_start, t_end, params.n_steps)

    def simulate(self, f_updates=None):
        for i, t in enumerate(self.times):

            # If first timestep, initialize the positions and velocities from the given conditions
            if i == 0:
                self.posns[i, :, :] = self.params.posns_0
                self.vels[i, :, :] = self.params.vels_0
                continue
            
            last_posns = self.posns[i-1, :, :]
            last_vels = self.vels[i-1, :, :]
            
            self.posns[i, :, :], self.vels[i, :, :] = self.step(t, last_posns, last_vels)

            if f_updates is not None:
                f_updates(self, i, t)

    def step(self, t, last_posns, last_vels):
        # 1. Preallocate posn matrix and naively propagate forward last velocity
        next_posns = np.zeros(last_posns.shape)
        next_vels = np.copy(last_vels)
        next_nudges = np.zeros(last_posns.shape)

        # 2. Now check for collisions: which velocities do we update?
        # Purely narrow-phase collision checking: Check every possible particle pair
        # TODO: Seperate out broad-phase from narrow phase by using KD or Quad trees.

        # Preallocate list of contact normals, which we will populate as we detect collisions
        # This way we only calculate once for each pair
        collision_pairs = set()

        for i_particle, particle in enumerate(last_posns.T):
            # Get distance between current particle and all other particles
            particle_as_col = np.atleast_2d(particle).T 
            distances = np.linalg.norm(particle_as_col - last_posns, axis=0)

            # Check if distances between current particle and any other ones are within the collision threshold
            # TODO: Elementwise compare against a list of expected collision distances for each particle based on particle.particle_r + collision_particle.particle_r
            colliding = (distances <= self.params.particles_r[i_particle] * 2) 
            colliding[i_particle] = False

            # Check if particle is still in box
            # Results in a list of if each component of the particle position is outside of the box bounds
            box_max = 0.5 * np.array([self.params.box_dims]).T
            box_min = -box_max
            colliding_with_box = (particle_as_col >= box_max) | (particle_as_col <= box_min)

            if np.any(colliding_with_box):
                if self.params.print_debug: print(f"Collision between {i_particle} and box happened at time {t}")
                # Negate the velocity in the direction which encountered a wall
                # IE, if encountering a wall for maximum Y bounds, then flip the Y velocity
                last_vel = last_vels[:, i_particle]
                
                # Create a mask which corresponds to the offending direction
                # If encountering maximum Y bounds, then this should be [0, 1, 0]
                collision_dirs_mask = colliding_with_box.astype(np.int32).squeeze()
                flip_matrix = np.diag(-collision_dirs_mask) # Diagonalize the negative mask to create a matrix which will flip the correct velocity when multiplied with velocity

                # Apply the flip via multiplication
                next_vels[:, i_particle] = flip_matrix @ last_vel
            elif np.any(colliding):
                # Find the closest particle - 
                # NOTE: Hopefully choosing the minimum-distance collision will also auto-resolve any multiple-collision scenario
                try:
                    min_nonzero_dist = np.min(distances[np.nonzero(distances)])
                except:
                    breakpoint()
                i_collision_particle = int(np.argwhere(distances == min_nonzero_dist)[0])
                collision_pair = frozenset((i_particle, i_collision_particle))

                # This collision has been handled already
                # Note that since set-retrieval is O(1) via hashing, this is faster than just simply setting the velocities twice.
                if collision_pair in collision_pairs:
                    continue

                collision_pairs.add(collision_pair)
                if self.params.print_debug: print(f"Collision between {i_particle} and {i_collision_particle} happened at time {t}")
                
                ## Implement general linear collision handling:
                # See https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects

                # Pull associated quantities with each particle
                posn_1, posn_2 = last_posns.T[[i_particle, i_collision_particle]]
                vel_1, vel_2 = last_vels.T[[i_particle, i_collision_particle]]
                m_1, m_2 = self.params.particles_mass[[i_particle, i_collision_particle]]

                # Find the contact normals: (x_1 - x_2) or (x_2 - x_1)
                # By "contact normal" I mean the vector between the center of mass of each object, which is *normal* to the contact surface. 
                # normal_1 points from the center of particle_1 *away* from particle_2
                normal_1 = posn_1 - posn_2                                              # Normal vector centered on "particle"
                normal_1 = normal_1 / np.linalg.norm(normal_1)                          # Normalize the vector to unit length
                normal_2 = -normal_1                                                    # Normal vector centered on "collision_particle"

                # Find the normal components of the original velocities of each particle
                # This is the component that gets "flipped" by the collision as it's on the line of force. The rest of the velocity is untouched.
                # Thus we will scale and subtract this from the original 
                vel_1_normal = np.dot(vel_1 - vel_2, normal_1) * normal_1
                vel_2_normal = np.dot(vel_2 - vel_1, normal_2) * normal_2

                vel_1_new = vel_1 - 2 * m_2 / (m_1 + m_2) * vel_1_normal
                vel_2_new = vel_2 - 2 * m_1 / (m_1 + m_2) * vel_2_normal

                # Finally, resolve particles caught in overlapping position by moving them to be tangent to each other.
                overlap_distance = self.params.particles_r[i_particle] * 2 - (np.linalg.norm(posn_1 - posn_2))
                nudge_1_new = overlap_distance * normal_1
                nudge_2_new = overlap_distance * normal_2
                
                # Save all the values
                next_vels[:, i_particle] = vel_1_new
                next_vels[:, i_collision_particle] = vel_2_new

                next_nudges[:, i_particle] += nudge_1_new
                next_nudges[:, i_collision_particle] += nudge_2_new


        # 3. Now that we have updated all velocities accordingly, let's integrate the velocities to find the new positions
        # TODO: For entities where a collision after the next step is anticipated, we need to do special things to prevent ghosting
        # r_next = r_current + vel * dt
        next_posns = last_posns + next_vels * self.params.dt + next_nudges

        ## 4. Resolve particles caught in illegal positions
        # Ensure particles are within box
        next_posns = np.clip(next_posns, box_min, box_max)

        return next_posns, next_vels
        