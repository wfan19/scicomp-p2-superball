import pygame as pg
import moderngl as mgl

import numpy as np

import graphics_utils

def make_sphere(r, longs, lats):
    vertices = np.zeros((longs*lats, 3))
    uv = np.zeros((longs*lats, 2))
    normals = np.zeros((longs*lats, 3))

    u = np.linspace(0, np.pi, lats)
    v = np.linspace(0, 2*np.pi, longs)

    ## Generate list of all vertices (points) on the sphere's geometry
    # This is such an un-pythonic way of doing this but oh well
    i = 0
    for u_i in u:
        for v_i in v:
            # Parameteric equation for a circle
            vertices[i] = np.array([
                r*np.sin(u_i)*np.cos(v_i),
                r*np.sin(u_i)*np.sin(v_i),
                r*np.cos(u_i)
            ])

            # Save vector normal to the mesh surface (needed for lighting)
            # The vector normal to the sphere's surface is the normalized version of the position itself.
            normals[i] = 1/r * vertices[i]

            # Save points for coordinate grid
            # Our parameterization is a coordinate grid itself, so we can just save it
            uv[i] = np.array([
                u_i, v_i
            ])

            i += 1

    ## Save indices of vertex-triplets which each correspond to a single triangle face.
    # Each set of grid-cell spanned by (u_i, v_i) and (u_i+1, v_i+1) contains two triangles
    indices = np.zeros((lats * longs * 2, 3))
    i = 0
    for u_i in range(lats - 1):
        for v_i in range(longs - 1):
            # For each grid cell with (u_i, v_i) at the bottom there are two triangles

            # First we do the "left-side" triangle
            indices[i] = np.array([
                u_i * longs         + v_i,          # Bottom left corner of grid cell
                (u_i + 1) * longs   + (v_i + 1),    # Top right corner of grid cell (reach to next row)
                (u_i) * longs   + (v_i + 1),        # Bottom right corner of grid cell
            ])

            # Next the "right-side" triangle
            indices[i+1] = np.array([
                u_i * longs         + v_i,          # Bottom left corner
                (u_i + 1) * longs   + v_i,          # Top left corner
                (u_i + 1) * longs   + (v_i + 1)     # Top right corner
            ])

            i += 2

    return vertices, uv, normals

vertex_shader = '''
    #version 330

    layout (location=0) in vec3 in_position;
    uniform mat4 mat_projection;
    uniform mat4 mat_view;  // Matrix representing camera_T_camera_world
    uniform mat4 mat_model; // Matrix representing world_T_world_model

    void main() {
        gl_Position = mat_projection * mat_view * mat_model * vec4(in_position, 1.0);
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

    # Initialize camera
    # TODO: Unit testing for all the glm re-implementation functions!
    mat_projection = graphics_utils.mat_projection(np.deg2rad(50), display[0]/display[1], 0.1, 100)
    mat_view = graphics_utils.lookAt(np.array([2, 3, 3]), np.array([0, 0, 0]), np.array([0, 1, 0]))

    clock = pg.time.Clock()

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

    shader_program['mat_projection'].write(mat_projection)
    shader_program['mat_view'].write(mat_view)
    shader_program['mat_model'].write(np.eye(4, dtype="float32"))

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                shader_program.release()
                
                cube_vbo.release()
                cube_vao.release()

                pg.quit()
                quit()
        
        context.clear(color=(0.08, 0.16, 0.18))

        t = pg.time.get_ticks() * 0.001
        mat_cube_pose = graphics_utils.translate(np.cos(t), 0, np.sin(t))
        shader_program['mat_model'].write(mat_cube_pose)

        cube_vao.render()

        pg.display.flip()
        clock.tick(60)