import numpy as np
import matplotlib.pylab as plt
from scipy.stats import pearsonr
import pandas as pd
from random import random, randint
from time import perf_counter as timer

def read_mesh(filename):
    #vertices: np array with the coordinates of each vertex
    #faces: np array with the three vertices that forms the face
    #conexions: set of faces around each vertex
    mesh = open(filename)
    n, m = mesh.readline().split()
    n = int(n)
    m = int(m)
    vertices = []
    faces = []
    conexions = [set() for i in range(n)]

    for i in range(n):
        vertices.append(mesh.readline().split())
        for j in range(3):
            vertices[i][j] = float(vertices[i][j])

    for i in range(m):
        faces.append(mesh.readline().split())
        for j in range(3):
            faces[i][j] = int(faces[i][j]) - 1
            conexions[faces[i][j]].add(i)

    return np.array(vertices), np.array(faces) , conexions

def get_normal(vertices, faces):
    # Gets the normal vector for all the faces
    normal = []
    area = []
    for face in faces:
        v1 = vertices[face[1]] - vertices[face[0]]
        v2 = vertices[face[2]] - vertices[face[0]]
        n = np.cross(v1, v2)
        a = np.sqrt(np.dot(n, n))
        area.append(a)
        normal.append(n / a)
    return np.array(normal), np.array(area)

def simple_box_ray_intersect( triangle , p , d ):
    #b: BB max point
    #r(t) = p + td
    z = np.max(triangle[:, 2])

    t = (max(triangle[:, 2] - p[2]) / d[2])

    if d[0] >= 0 : x = p[0] + t*d[0] < np.max(triangle[:, 0])
    else : x = p[0] + t*d[0] > np.min(triangle[:, 0])

    if d[1] >= 0 : y = p[1] + t*d[1] < np.max(triangle[:, 1])
    else : y = p[1] + t*d[1] > np.min(triangle[:, 1])

    if x and y : return True
    else : return False

def read_extra_data(filename):
    # Reads preprocessed data, adjacency matrix and BB boxes
    adjacency = []
    adjacency_file = open("input/preprocessing/adjacency_" + filename + ".txt", "r")
    line = adjacency_file.readline()
    while line:
        adjacency.append(line.split())
        for i in range(len(adjacency[-1])): adjacency[-1][i] = int(adjacency[-1][i])
        line = adjacency_file.readline()
    adjacency_file.close()

    BB = []
    BB_file = open("input/preprocessing/BB_" + filename + ".txt", "r")
    line = BB_file.readline()
    while line:
        BB.append(line.split())
        for i in range(6): BB[-1][i] = float(BB[-1][i])
        line = BB_file.readline()
    BB_file.close()
    BB = np.array(BB)

    return adjacency, BB.reshape((len(BB), 2, 3))

def Moller_Trumbore(triangle, p, d, n=False, r=0):
    # gets the coordinates of the intersection of r(t) = p+t*d and the triangle, T(alpha,beta) = alpha*v1 + beta*v2
    v1 = triangle[1] - triangle[0]
    v2 = triangle[2] - triangle[0]
    if not n:  # computes normal vector
        n = np.cross(v1, v2)
        n = n / np.sqrt(np.dot(n, n))
    p = p - n * r

    M = np.concatenate((-d, v1, v2)).reshape(3, 3).transpose()
    sol = np.linalg.solve(M, p - triangle[0])
    return sol[0], sol[1], sol[2]

def cylinder_mesh_intersection( p , d , vertices ):
    d = d/np.sqrt(np.dot(d,d))
    x = np.subtract(vertices,p)
    b = np.dot( x , d )
    return sum((x*x).transpose()) - b*b

def particle_mesh_intersection( p , d , r ,vertices , faces , conexions ):
    candidates = np.arange(len(vertices))[ r*r-4 < cylinder_mesh_intersection( p , d , vertices ) < r*r+4 ]
    intersection = False
    t_intersection = 1000
    for candidate in candidates:
        for triangle in conexions[candidate]:
            t , alpha , beta = Moller_Trumbore( vertices[faces[triangle]] , p , d , False , r )
            if alpha > 0 and beta > 0 and alpha + beta < 1 and t < t_intersection:
                intersection = triangle
                t_intersection = t

    return intersection

def mesh_characterization(phi, theta , r , a , b , vertices , faces , conexions  ):
    normals_file = open("output_part/normals_" + str(r) + "_" + str(phi) + "_" + str(theta) + ".txt", "w")
    normal, _ = get_normal(vertices, faces)
    if b > len(faces): b = len(faces)
    d = np.array([-np.sin(phi / 180 * np.pi) * np.cos(theta / 180 * np.pi),
                  -np.sin(phi / 180 * np.pi) * np.sin(theta / 180 * np.pi),
                  np.cos(phi / 180 * np.pi)])
    for face in range(a, b):
        incidence = sum(vertices[faces[face]]) / 3
        face_i = particle_mesh_intersection(incidence, d, r , vertices, faces, conexions )
        print( face , face_i )
        if face_i:
            print( "ok" )
            normals_file.write(str(face_i) + "\n")
    normals_file.close()

if __name__ == "__main__":
    vertices , faces , conexions = read_mesh("input/mucus_large.txt")
    adjacency, BB = read_extra_data("large")
    mesh_characterization(60, 0, 50, 0, 400000000, vertices, faces, conexions )

