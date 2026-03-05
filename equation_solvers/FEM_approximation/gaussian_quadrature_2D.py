import numpy as np
from .parse_mesh import *

def reference_triangle(exactness_degree):  

    if exactness_degree == 1:
        nodes = np.array([[1/3, 1/3]])
        weights = np.array([1.0])

    return nodes, weights

def integrate_on_triangle(f, vertices, exactness_degree):

    nodes, weights = reference_triangle(exactness_degree)

    # Unpack triangle vertices
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]

    # Jacobian determinant
    J = abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
    area = J / 2

    triangle_estimate = 0.0

    for (xi, eta), w in zip(nodes, weights):
        # map from reference triangle
        x = x1 + xi*(x2-x1) + eta*(x3-x1)
        y = y1 + xi*(y2-y1) + eta*(y3-y1)

        triangle_estimate += w * f(x, y)

    return triangle_estimate * area

def integrate_mesh(f, mesh_vertices, mesh_elements, exactness_degree):
    total_estimate = 0.0

    for elem in mesh_elements:
        vertices = mesh_vertices[elem]
        total_estimate += integrate_on_triangle(f, vertices, exactness_degree)
    return total

