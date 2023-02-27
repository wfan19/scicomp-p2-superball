from typing import List

import numpy as np

import moderngl as mgl
import pygame as pg
import graphics_utils

from ColloidSim import ColloidSim, ColloidSimParams

class Camera():
    mat_view:           np.ndarray
    mat_proj:           np.ndarray

    position:           np.ndarray = np.array([0.5, 4, 1.5], dtype=np.float32)
    forward:            np.ndarray = np.array([1, 0, 0], dtype=np.float32)
    right:              np.ndarray = np.array([0, 1, 0], dtype=np.float32)
    up:                 np.ndarray = np.array([0, 0, 1], dtype=np.float32)
    pitch:              float = 0
    yaw:                float = 45

    def __init__(self, window_size):
        self.mat_proj = graphics_utils.mat_projection(np.deg2rad(50), window_size[0]/window_size[1], 0.1, 100)
        self.mat_view = graphics_utils.lookAt(self.position, np.array([0, 0, 0]), np.array([0, 0, 1]))

    def update(self, dt):
        # Move camera based on keyboard input
        CAMERA_SPEED = 0.002
        step_size = CAMERA_SPEED * dt
        keys = pg.key.get_pressed()
        if keys[pg.K_w]:
            self.position += self.forward * step_size
        if keys[pg.K_s]:
            self.position -= self.forward * step_size
        if keys[pg.K_a]:
            self.position -= -self.right * step_size # This is a hack, should be positive
        if keys[pg.K_d]:
            self.position += -self.right * step_size # This is a hack, should be positive
        if keys[pg.K_q]:
            self.position += self.up * step_size
        if keys[pg.K_e]:
            self.position -= self.up * step_size

        # Rotate camera based on mouse movement
        SENSITIVITY = 0.005
        rel_x, rel_y = pg.mouse.get_rel()
        self.yaw += rel_x * SENSITIVITY
        self.pitch -= rel_y * SENSITIVITY
        self.pitch = max(-89, min(89, self.pitch))

        # Recompute local camera vectors
        # Pitch and Yaw give us the forward vector via spherical coordinates
        self.forward = np.array([
            np.cos(self.yaw) * np.cos(self.pitch),
            np.sin(self.yaw) * np.cos(self.pitch),
            np.sin(self.pitch)
        ])
        self.forward = self.forward / np.linalg.norm(self.forward)

        self.right = np.cross(self.forward, np.array([0, 0, 1]))
        self.right = self.right / np.linalg.norm(self.right)

        self.up = np.cross(self.right, self.forward)
        self.up = self.up / np.linalg.norm(self.up)

        self.mat_view = graphics_utils.lookAt(self.position, self.position + self.forward, self.up)
        

class VizSphere():
    program:                mgl.Program
    vertex_array:           mgl.VertexArray

    camera:                 Camera

    def __init__(self, context, camera, r):
        with open("default.vert") as file:
            vertex_shader = file.read()
        with open("default.frag") as file:
            fragment_shader = file.read()

        self.program = context.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        self.camera = camera

        # Initialize camera
        # TODO: These first two should be shared across all objects, instead of needing to copy them per-object.

        self.program['mat_projection'].write(self.camera.mat_proj)
        self.program['mat_view'].write(self.camera.mat_view)
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
        self.program['mat_projection'].write(self.camera.mat_proj)
        self.program['mat_view'].write(self.camera.mat_view)
        self.vertex_array.render()

class ColloidViz():
    sim:        ColloidSim

    context:    mgl.Context
    spheres:    List[VizSphere]

    box_program:    mgl.Program
    box_vao:        mgl.vertex_array

    camera:     Camera

    clock:      pg.time.Clock

    def __init__(self, colloid_sim: ColloidSim, window_size = (1200, 900)):
        self.sim = colloid_sim

        # Set up the Pygame window
        pg.init()
        self.clock = pg.time.Clock()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.set_mode(window_size, pg.DOUBLEBUF | pg.OPENGL)

        ## Initialize Opengl context
        self.context = mgl.create_context()
        self.context.enable(flags=mgl.DEPTH_TEST)
        
        self.camera = Camera(window_size)

        ## Create the individual sphere meshes and buffer them into GPU memory
        self.spheres = []
        for r in colloid_sim.params.particles_r:
            self.spheres.append(VizSphere(self.context, self.camera, r))

        ## Render the boundary box
        # TODO: This is a lot of code - should be moved elsewhere
        with open("default.vert") as file:
            vertex_shader = file.read()
        with open("default.frag") as file:
            fragment_shader = file.read()
        self.box_program = self.context.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        self.box_program['mat_model'].write(graphics_utils.translate(0, 0, 0))
        self.box_program['mat_projection'].write(self.camera.mat_proj)
        self.box_program['mat_view'].write(self.camera.mat_view)

        # Build the outsidie boundary cube
        # box_vertices = graphics_utils.make_cube(self.sim.params.box_dims[0])
        l_x, l_y, l_z = self.sim.params.box_dims
        box_vertices = graphics_utils.make_wireframe_cube(l_x, l_y, l_z)
        box_vbo = self.context.buffer(box_vertices)
        self.box_vao = self.context.vertex_array(self.box_program, [(box_vbo, '3f', "in_position")], mode=mgl.LINES)
    
    def visualize(self, camera_posn= np.array([0.5, 4, 1.5]), control_camera=False):
        if control_camera:
            pg.mouse.set_visible(False)
            pg.event.set_grab(True)

        self.camera.position = camera_posn
        self.camera.mat_view = graphics_utils.lookAt(camera_posn, np.array([0, 0, 0]), np.array([0, 0, 1]))

        for positions in self.sim.posns:
            # First detect if there are exit conditions
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE): 
                    # If window is closed, shutdown program
                    for sphere in self.spheres:
                        sphere.program.release()
                        sphere.vertex_array.release()

                    # Exit program
                    pg.quit()
                    quit()
            
            dt = self.clock.tick(60)
            self.context.clear(color=(0.08, 0.16, 0.18))

            if control_camera:
                self.camera.update(dt)

            for i, sphere in enumerate(self.spheres):
                sphere.draw(positions.T[i])

            self.box_program["mat_view"].write(self.camera.mat_view)
            self.box_vao.render()

            pg.display.flip()