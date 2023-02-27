import pygame as pg
import moderngl as mgl

import numpy as np

import graphics_utils

if __name__ == "__main__":
    pg.init()
    display = (800, 600)

    pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
    pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
    pg.display.set_mode(display, pg.DOUBLEBUF | pg.OPENGL)

    context = mgl.create_context()
    context.enable(flags=mgl.DEPTH_TEST)

    with open("default.vert") as file:
        vertex_shader = file.read()
    with open("default.frag") as file:
        fragment_shader = file.read()

    shader_program = context.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

    # Initialize camera
    # TODO: Unit testing for all the glm re-implementation functions!
    mat_projection = graphics_utils.mat_projection(np.deg2rad(50), display[0]/display[1], 0.1, 100)
    mat_view = graphics_utils.lookAt(np.array([2, 3, 3]), np.array([0, 0, 0]), np.array([0, 0, 1]))

    clock = pg.time.Clock()

    # cube_vertices = np.array([
    #     [-1, -1, 1],
    #     [1, -1, 1],
    #     [1, 1, 1],
    #     [-1, 1, 1],
    #     [-1, 1, -1],
    #     [-1, -1, -1],
    #     [1, -1, -1],
    #     [1, 1, -1]
    # ], dtype="float32")
    # cube_triangles = [
    #     [0, 2, 3], [0, 1, 2],
    #     [1, 7, 2], [1, 6, 7],
    #     [6, 5, 4], [4, 7, 6],
    #     [3, 4, 5], [3, 5, 0],
    #     [3, 7, 4], [3, 2, 7],
    #     [0, 6, 1], [0, 5, 6],
    # ]
    # cube_vertex_data = np.array([cube_vertices[i_vertex] for triangle in cube_triangles for i_vertex in triangle])
    # cube_vbo = context.buffer(cube_vertex_data)
    # cube_vao = context.vertex_array(shader_program, [(cube_vbo, '3f', 'in_position')])

    sphere_vertices, sphere_normals, sphere_uv, sphere_indices = graphics_utils.make_sphere(1, 30, 30)
    sphere_vbo  = context.buffer(sphere_vertices)
    uv_vbo      = context.buffer(sphere_uv)
    indices_vbo = context.buffer(sphere_indices)

    sphere_vao = context.vertex_array(shader_program, [
            (sphere_vbo, '3f', 'in_position'),
            (uv_vbo, '2f', 'in_textcoord')
        ],
        index_buffer=indices_vbo,
        index_element_size=4
    )

    shader_program['mat_projection'].write(mat_projection)
    shader_program['mat_view'].write(mat_view)
    shader_program['mat_model'].write(np.eye(4, dtype="float32"))

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                shader_program.release()

                sphere_vbo.release()
                sphere_vao.release()
                
                # cube_vbo.release()
                # cube_vao.release()

                pg.quit()
                quit()
        
        context.clear(color=(0.08, 0.16, 0.18))

        t = pg.time.get_ticks() * 0.001
        mat_cube_pose = graphics_utils.rotate(0, 0, 1, t) @ graphics_utils.translate(np.cos(t), np.sin(t), t * 0.25) 
        shader_program['mat_model'].write(mat_cube_pose)

        # cube_vao.render()
        sphere_vao.render()

        pg.display.flip()
        clock.tick(60)