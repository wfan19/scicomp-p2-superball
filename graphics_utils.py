import numpy as np

def mat_projection(fov, aspect, near, far, dtype=np.float32):
    """ Generation of a projective transform matrix based on camera parameters.
    See https://stackoverflow.com/questions/53245632/general-formula-for-perspective-projection-matrix
    and also https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml

    Args:
        fov (float): Field of view in radians
        aspect (float): Aspect ratio of camera view
        near (float): Close clipping distance
        far (float): Far clipping distance
    """
    
    f = 1/(np.tan(fov / 2))
    out = np.array([
        [1/aspect * f, 0, 0, 0],
        [0           , f, 0, 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0]
    ], dtype=dtype)

    return np.ascontiguousarray(out.T)

def lookAt(eye_posn: np.ndarray, center: np.ndarray, up: np.ndarray, dtype=np.float32):
    """ Generate homogenous transform matrix for camera-world transform, given desired camera position and view angle
    See https://github.com/g-truc/glm/blob/b3f87720261d623986f164b2a7f6a0a938430271/glm/ext/matrix_transform.inl#L99 for reference.

    Args:
        eye_posn (np.ndarray): Desired camera center position
        center (np.ndarray): Point to center in camera view
        up (np.ndarray): Which way the camera's up should be in the world frame
        dtype (np.dtype): _description_. Defaults to np.float32.

    Returns:
        np.ndarray: 4x4 homogenous transform matrix representing camera_T_camera_world: Transformation from camera center to world center in the camera frame.
    """
    f = center - eye_posn 
    f = f / np.linalg.norm(f)

    s = np.cross(up, f)
    s = s / np.linalg.norm(s)

    u = np.cross(f, s)

    posn = np.array([
        -s.flatten(),
        -u.flatten(),
        f.flatten()
    ]) @ eye_posn

    out = np.eye(4, dtype=dtype)
    out[0:3, 0:3] = np.array([-s, u, -f])
    out[0:3, 3] = posn

    return np.ascontiguousarray(out.T)

def rotate(x, y, z, theta, dtype=np.float32):
    """ Generate a pure-rotation 4x4 homogenous transform matrix, using axis-angle as input

    Args:
        x (float): x-component of rotation axis
        y (float): y-component of rotation axis
        z (float): z-component of rotation axis
        theta (float): angle of rotation
        dtype (np.dtype): _description_. Defaults to np.float32.
    """

    axis = np.array([x, y, z])
    axis = axis / np.linalg.norm(axis)

    mat_skew_sym = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    # Rodrigues' formula: the analytic exponential map on the group of 3x3 rotation matrices
    # Also known as the conversion from axis-angle to rotation matrix representations for rotations
    R = np.eye(3) + np.sin(theta) * mat_skew_sym + (1 - np.cos(theta)) * (mat_skew_sym @ mat_skew_sym)

    out = np.eye(4)
    out[0:3, 0:3] = R

    return np.ascontiguousarray(out.astype(dtype).T)

def translate(x, y, z, dtype=np.float32):
    """ Generate a pure-translation 4x4 homogenous transform matrix
    Reference: See "3D Exponential Map" under https://thenumb.at/Exponential-Rotations/

    Args:
        x (float): translation in x direction
        y (float): translation in y direction
        z (float): translation in z direction
        dtype (np.dtype): _description_. Defaults to np.float32.

    Returns:
        np.ndarray: 4x4 Homogenous transform matrix
    """
    out = np.eye(4, dtype=dtype)
    out[0:3, 3] = np.array([x, y, z])

    return np.ascontiguousarray(out.T)

def make_sphere(r, longs, lats):
    """ Generates the geometry (vertices, normals, uvs, indices) for a sphere mesh.

    Args:
        r (float): Radius of sphere
        longs (int): Number of longitude lines
        lats (int): Number of latitude lines

    Returns:
        np.1dArray: vertices - the list of vertices in 3d position, vectorized
        np.1dArray: normals - the list of surface normal vectors in 3d position, vectorized
        np.1dArray: uv - the list of uv coordinates in 2d position, vectorized
        np.1dArray: indices - the list of indices of vertex points which form individual triangular faces tilling the mesh
    """
    vertices = np.zeros((longs*lats, 3), dtype=np.float32)
    uv = np.zeros((longs*lats, 2), dtype=np.float32)
    normals = np.zeros((longs*lats, 3), dtype=np.float32)
    indices = np.zeros((lats * longs * 2, 3), dtype=np.int32)

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

    return vertices.flatten(), normals.flatten(), uv.flatten(), indices.flatten()

def make_cube(l):
    # TODO: Support for different xyz dimensions
    cube_vertices = l/2 * np.array([
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1]
    ], dtype=np.float32)
    cube_triangles = [
        [0, 2, 3], [0, 1, 2],
        [1, 7, 2], [1, 6, 7],
        [6, 5, 4], [4, 7, 6],
        [3, 4, 5], [3, 5, 0],
        [3, 7, 4], [3, 2, 7],
        [0, 6, 1], [0, 5, 6],
    ]
    cube_vertex_data = np.array([cube_vertices[i_vertex] for triangle in cube_triangles for i_vertex in triangle])
    
    # TODO: Cube UV and normals would be nice but not necessary since for now we're just using it for the box

    return cube_vertex_data

def make_wireframe_cube(x, y, z):
    box_vertices = 0.5 * np.array([
        # First the bottom four edges
        [-x, -y, -z], [-x, y, -z],
        [-x, y, -z], [x, y, -z],
        [x, y, -z], [x, -y, -z],
        [x, -y, -z], [-x, -y, -z],
        
        # Next the top four
        [-x, -y, z], [-x, y, z],
        [-x, y, z], [x, y, z],
        [x, y, z], [x, -y, z],
        [x, -y, z], [-x, -y, z],

        # Next the four vertical ones
        [-x, -y, -z], [-x, -y, z],
        [x, -y, -z], [x, -y, z],
        [-x, y, -z], [-x, y, z],
        [x, y, -z], [x, y, z],
    ], dtype=np.float32).flatten()
    return box_vertices