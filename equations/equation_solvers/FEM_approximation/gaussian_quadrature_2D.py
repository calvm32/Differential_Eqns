import numpy as np
from equation_solvers.FEM_approximation.parse_mesh import *
from pathlib import Path

def reference_triangle(exactness_degree): 

    if exactness_degree == 1:
        nodes = np.array([[1/3, 1/3]])
        weights = np.array([1.0])
    elif exactness_degree == 2:
        nodes = np.array([[0.211324865, 0.166666667],[0.211324865, 0.622008467], [0.788675134, 0.044658198], [0.788675134, 0.166666667]])
        weights = np.array([0.197168783, 0.197168783, 0.052831216, 0.052831216])
    elif exactness_degree == 3:
        nodes = np.array([[0.112701665, 0.100000000],[0.112701665, 0.443649167], [0.112701665, 0.787298334], [0.500000000, 0.056350832], 
                         [0.500000000, 0.250000000],[0.500000000, 0.443649167], [0.887298334, 0.012701665], [0.887298334, 0.056350832], [0.887298334, 0.100000000]])
        weights = np.array([0.068464377, 0.109543004, 0.068464377, 0.061728395,
                           0.098765432, 0.061728395, 0.008696116, 0.013913785, 0.008696116])       
    else:
        print("\n Warning!!!! must enter exactness degree 1, 2, or 3\n") 

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

def integrate_mesh(f, mesh_nodes, mesh_elements, exactness_degree):
    total_estimate = 0.0

    for elem in mesh_elements:
        vertices = [mesh_nodes[elem[0]-1], mesh_nodes[elem[1]-1], mesh_nodes[elem[2]-1]] 
        total_estimate += integrate_on_triangle(f, vertices, exactness_degree)

    return total_estimate

def main():

    mesh_path = Path(__file__).parent.parent.parent / "meshes" / "L1.msh"

    f = lambda x, y: np.sin(x)*np.cos(y)
    # on the recangle [a,b]x[c,d], this evaluates to exactly (cos(a)-cos(b))*(cos(c)-cos(d))
    # on the L-shaped domain, we have [0,1]x[0,0.5] + [0,0.5]x[0.5,1]

    print()
    total_exact = (-1)*(np.cos(1) - np.cos(0))*(np.sin(0.5) - np.sin(0)) + (-1)*(np.cos(0.5) - np.cos(0))*(np.sin(1) - np.sin(0.5))

    mesh_nodes, mesh_elements = parse_mesh(mesh_path)
    total_estimate = integrate_mesh(f, mesh_nodes, mesh_elements, 1)

    print(f"\n\nEstimated integrand: {total_estimate:.4f}")
    print(f"Exact integrand: {total_exact:.4f}\n\n")

if __name__=="__main__":
    main()