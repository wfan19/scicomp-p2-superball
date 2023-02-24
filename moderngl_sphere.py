import pygame as pg
import moderngl as mgl

import numpy as np

vertex_shader = '''
    #version 330

    layout (location=0) in vec3 in_position;

    void main() {
        gl_Position = vec4(in_position, 1.0);
    }
'''

fragment_shader = '''
    #version 330

    layout (location=0) out vec4 out_color;

    void main() {
        vec3 color = vec3(1, 0, 0);
        out_color = vec4(color, 1.0);
    }
'''

if __name__ == "__main__":
    pg.init()
    display = (800, 600)

    pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
    pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
    pg.display.set_mode(display, pg.DOUBLEBUF | pg.OPENGL)

    context = mgl.create_context()
    shader_program = context.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

    clock = pg.time.Clock()
        
    triangle_vertices = np.array([(-0.6, -0.8, 0), (0.6, -0.8, 0), (0, 0.8, 0)], dtype="float32")
    triangle_vbo = context.buffer(triangle_vertices)
    triangle_vao = context.vertex_array(shader_program, [(triangle_vbo, '3f', 'in_position')])

    cube_vertices = np.array([
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1]
    ], dtype="float32")
    cube_triangles = [
        [0, 2, 3], [0, 1, 2],
        [1, 7, 2], [1, 6, 7],
        [6, 5, 4], [4, 7, 6],
        [3, 4, 5], [3, 5, 0],
        [3, 7, 4], [3, 2, 7],
        [0, 6, 1], [0, 5, 6],
    ]
    cube_vertex_data = np.array([cube_vertices[i_vertex] for triangle in cube_triangles for i_vertex in triangle])
    cube_vbo = context.buffer(cube_vertex_data)
    cube_vao = context.vertex_array(shader_program, [(cube_vbo, '3f', 'in_position')])

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                triangle_vbo.release()
                shader_program.release()
                triangle_vao.release()

                pg.quit()
                quit()
        
        context.clear(color=(0.08, 0.16, 0.18))

        # triangle_vao.render()
        cube_vao.render()

        pg.display.flip()

        clock.tick(60)
