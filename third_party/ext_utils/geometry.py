"""
Geometry stuff.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def triangle_direction_intersection(tri, trg):
    '''
    Finds where an origin-centered ray going in direction trg intersects a triangle.
    Args:
        tri: 3 X 3 vertex locations. tri[0, :] is 0th vertex.
    Returns:
        alpha, beta, gamma
    '''
    p0 = np.copy(tri[0, :])
    # Don't normalize
    d1 = np.copy(tri[1, :]) - p0;
    d2 = np.copy(tri[2, :]) - p0;
    d = trg / np.linalg.norm(trg)

    mat = np.stack([d1, d2, d], axis=1)

    try:
      inv_mat = np.linalg.inv(mat)
    except np.linalg.LinAlgError:
      return False, 0

    # inv_mat = np.linalg.inv(mat)
    
    a_b_mg = -1*np.matmul(inv_mat, p0)
    is_valid = (a_b_mg[0] >= 0) and (a_b_mg[1] >= 0) and ((a_b_mg[0] + a_b_mg[1]) <= 1) and (a_b_mg[2] < 0)
    if is_valid:
        return True, -a_b_mg[2]*d
    else:
        return False, 0


def project_verts_on_mesh(verts, mesh_verts, mesh_faces):
    verts_out = np.copy(verts)
    for nv in range(verts.shape[0]):
        max_norm = 0
        vert = np.copy(verts_out[nv, :])
        for f in range(mesh_faces.shape[0]):
            face = mesh_faces[f]
            tri = mesh_verts[face, :]
            # is_v=True if it does intersect and returns the point
            is_v, pt = triangle_direction_intersection(tri, vert)
            # Take the furthest away intersection point
            if is_v and np.linalg.norm(pt) > max_norm:
                max_norm = np.linalg.norm(pt)
                verts_out[nv, :] = pt

    return verts_out


def R_2vect(vector_orig, vector_fin):
    """Calculate the rotation matrix required to rotate from one vector to another.
    For the rotation of one vector to another, there are an infinit series of rotation matrices
    possible.  Due to axially symmetry, the rotation axis can be any vector lying in the symmetry
    plane between the two vectors.  Hence the axis-angle convention will be used to construct the
    matrix with the rotation axis defined as the cross product of the two vectors.  The rotation
    angle is the arccosine of the dot product of the two unit vectors.
    Given a unit vector parallel to the rotation axis, w = [x, y, z] and the rotation angle a,
    the rotation matrix R is::
              |  1 + (1-cos(a))*(x*x-1)   -z*sin(a)+(1-cos(a))*x*y   y*sin(a)+(1-cos(a))*x*z |
        R  =  |  z*sin(a)+(1-cos(a))*x*y   1 + (1-cos(a))*(y*y-1)   -x*sin(a)+(1-cos(a))*y*z |
              | -y*sin(a)+(1-cos(a))*x*z   x*sin(a)+(1-cos(a))*y*z   1 + (1-cos(a))*(z*z-1)  |
    @param R:           The 3x3 rotation matrix to update.
    @type R:            3x3 numpy array
    @param vector_orig: The unrotated vector defined in the reference frame.
    @type vector_orig:  numpy array, len 3
    @param vector_fin:  The rotated vector defined in the reference frame.
    @type vector_fin:   numpy array, len 3
    """
    from math import acos, atan2, cos, pi, sin
    from numpy import array, cross, dot, float64, hypot, zeros
    from numpy.linalg import norm
    R = np.zeros((3,3))

    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / norm(vector_orig)
    vector_fin = vector_fin / norm(vector_fin)

    # The rotation axis (normalised).
    axis = cross(vector_orig, vector_fin)
    axis_len = norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle.
    angle = acos(dot(vector_orig, vector_fin))

    # Trig functions (only need to do this maths once!).
    ca = cos(angle)
    sa = sin(angle)
    # Calculate the rotation matrix elements.
    R[0,0] = 1.0 + (1.0 - ca)*(x**2 - 1.0)
    R[0,1] = -z*sa + (1.0 - ca)*x*y
    R[0,2] = y*sa + (1.0 - ca)*x*z
    R[1,0] = z*sa+(1.0 - ca)*x*y
    R[1,1] = 1.0 + (1.0 - ca)*(y**2 - 1.0)
    R[1,2] = -x*sa+(1.0 - ca)*y*z
    R[2,0] = -y*sa+(1.0 - ca)*x*z
    R[2,1] = x*sa+(1.0 - ca)*y*z
    R[2,2] = 1.0 + (1.0 - ca)*(z**2 - 1.0)
    return R
