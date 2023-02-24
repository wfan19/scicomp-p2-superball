import numpy as np

def mat_projection(fov, aspect, near, far, dtype="float32"):
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

def lookAt(eye_posn: np.ndarray, center: np.ndarray, up: np.ndarray, dtype="float32"):
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

def translate(x, y, z, dtype="float32"):
    out = np.eye(4, dtype=dtype)
    out[0:3, 3] = np.array([x, y, z])

    return np.ascontiguousarray(out.T)