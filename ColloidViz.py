from typing import List

import numpy as np

import moderngl as mgl
import pygame as pg
import graphics_utils

from ColloidSim import ColloidSim, ColloidSimParams

class VizSphere():
    
    program:                mgl.Program
    vertex_array:           mgl.VertexArray

    def __init__(self, context, r, mat_projection, mat_view):
        with open("default.vert") as file:
            vertex_shader = file.read()
        with open("default.frag") as file:
            fragment_shader = file.read()

        self.program = context.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

        # Initialize camera
        # TODO: These first two should be shared across all objects, instead of needing to copy them per-object.
        self.program['mat_projection'].write(mat_projection)
        self.program['mat_view'].write(mat_view)
        self.program['mat_model'].write(np.eye(4, dtype="float32"))

        sphere_vertices, sphere_normals, sphere_uv, sphere_indices = graphics_utils.make_sphere(r, 30, 30)
        sphere_vbo  = context.buffer(sphere_vertices)
        uv_vbo      = context.buffer(sphere_uv)
        indices_vbo = context.buffer(sphere_indices)

        self.vertex_array = context.vertex_array(self.program, [
                (sphere_vbo, '3f', 'in_position'),
                (uv_vbo, '2f', 'in_textcoord')
            ],
            index_buffer=indices_vbo,
            index_element_size=4
        )

    def draw(self, position: np.ndarray):
        self.program['mat_model'].write(graphics_utils.translate(position[0], position[1], position[2]))
        self.vertex_array.render()

class ColloidViz():
    sim:        ColloidSim

    context:    mgl.Context
    spheres:    List[VizSphere]

    clock:      pg.time.Clock

    def __init__(self, colloid_sim: ColloidSim, window_size = (1200, 900)):
        self.sim = colloid_sim

        pg.init()
        self.clock = pg.time.Clock()

        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.set_mode(window_size, pg.DOUBLEBUF | pg.OPENGL)

        self.context = mgl.create_context()
        self.context.enable(flags=mgl.DEPTH_TEST)
        
        mat_projection = graphics_utils.mat_projection(np.deg2rad(50), window_size[0]/window_size[1], 0.1, 100)
        mat_view = graphics_utils.lookAt(np.array([2, 3, 3]), np.array([0, 0, 0]), np.array([0, 0, 1]))

        self.spheres = []
        for r in colloid_sim.params.particles_r:
            self.spheres.append(VizSphere(self.context, r, mat_projection, mat_view))
    
    def visualize(self):
        for positions in self.sim.posns:
            # First detect if there are exit conditions
            for event in pg.event.get():
                if event.type == pg.QUIT: 
                    # If window is closed, shutdown program
                    for sphere in self.spheres:
                        sphere.program.release()
                        sphere.vertex_array.release()

                    # Exit program
                    pg.quit()
                    quit()
            
            self.context.clear(color=(0.08, 0.16, 0.18))
            t = pg.time.get_ticks() * 0.001

            for i, sphere in enumerate(self.spheres):
                sphere.draw(positions.T[i])

            pg.display.flip()
            self.clock.tick(60)