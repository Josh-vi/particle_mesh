import numpy as np
import matplotlib.pylab as plt
from scipy.stats import pearsonr
import pandas as pd
from random import random, randint
from time import perf_counter as timer


def read_mesh(filename):
    # vertices: np array with the coordinates of each vertex
    # faces: np array with the three vertices that forms the face
    # conexions: set of faces around each vertex
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

    return np.array(vertices), np.array(faces), conexions


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


def adjacency_graph(conexions, faces):
    # Get the set of adjecency faces around each face
    adjacency = []
    adjacency_file = open("input/preprocessing/adjacency_large.txt", "w")
    for f, face in enumerate(faces):
        v1 = conexions[face[0]]
        v2 = conexions[face[1]]
        v3 = conexions[face[2]]
        adjacency.append(v1.intersection(v2).union(v2.intersection(v3)).union(v3.intersection(v1)))
        adjacency[-1].remove(f)
        adjacency[-1] = list(adjacency[-1])
        adjacency_file.write(" ".join([str(adj) for adj in adjacency[-1]]) + "\n" )
    adjacency_file.close()
    return adjacency


def bounding_box(vertices, faces):
    BB = np.zeros((len(faces), 2, 3))
    # BB_file = open("input/preprocessing/BB_large.txt","w")
    for f, face in enumerate(faces):
        triangle = vertices[face]
        BB[f, 0, 0] = np.min(triangle[:, 0])
        BB[f, 0, 1] = np.min(triangle[:, 1])
        BB[f, 0, 2] = np.min(triangle[:, 2])
        BB[f, 1, 0] = np.max(triangle[:, 0])
        BB[f, 1, 1] = np.max(triangle[:, 1])
        BB[f, 1, 2] = np.max(triangle[:, 2])
        # BB_file.write(" ".join([str(b) for b in BB[f,0]]) + " " )
        # BB_file.write(" ".join([str(b) for b in BB[f,1]]) + "\n" )
    # BB_file.close()
    return BB


def starting_reference(vertices, conexions):
    # Gets list of vertices y and faces f in the line x = 0
    v_ref = 0
    starting_y = []
    starting_f = []
    while vertices[v_ref + 1][0] == 0:
        starting_y.append(vertices[v_ref + 1][1])
        starting_f.append(list(conexions[v_ref].intersection(conexions[v_ref + 1]))[0])
        v_ref += 1
    return np.array(starting_y), np.array(starting_f)


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


def spheric_angles(normal):
    # Converts
    x = np.array([1, 0, 0]);
    y = np.array([0, 1, 0]);
    z = np.array([0, 0, 1])
    phi = [];
    theta = []
    for n in normal:
        n = n / np.sqrt(np.dot(n, n))
        phi.append(np.arccos(np.dot(z, n)) / np.pi * 180)
        n[2] = 0;
        n = n / np.sqrt(np.dot(n, n))
        vx = np.dot(x, n);
        vy = np.dot(y, n)
        if vy >= 0:
            theta.append(np.arccos(vx) / np.pi * 180)
        else:
            theta.append(-np.arccos(vx) / np.pi * 180)
    return np.array(phi), np.array(theta)


def get_triangle(x, y, vertices, faces, adjacency):  # A* naive aproximation
    face = 1
    visited = {face}
    while not Moller_Trumbore_2d(x, y, vertices[faces[face]]):
        neighbors = vertices[faces[adjacency[face]]][:, :, 0:2]
        neighbors = [np.dot(np.array([x, y]) - sum(vertices[faces[face]][:, 0:2]) / 3,
                            sum(n) / 3 - sum(vertices[faces[face]][:, 0:2]) / 3) for n in neighbors]
        neighbors_sorted = [x for _, x in sorted(zip(neighbors, adjacency[face]), reverse=True)]
        for n in neighbors_sorted:
            if n not in visited:
                visited.add(n)
                face = n
                break
        else:
            return False
    return face


def bouncings(normal, x):
    bounces = []
    for n in normal:
        if np.dot(n, x) < 0:
            bounces.append(bounce(n, x))
            # print( n , x , bounces[-1])
    return np.array(bounces)


def bounce(n, x):
    # return elastic colission of x in n
    R = 2 * np.kron(n, n).reshape((3, 3)) - np.identity(3)
    return np.matmul(R, -x)


def simple_box_ray_intersect(b, p, d):
    # b: BB max point
    # r(t) = p + td
    t = (b[0] - p[0]) / d[0]
    return p[2] + t * d[2] < b[2]


def Moller_Trumbore_2d(x, y, triangle):
    v1 = triangle[1, 0:2] - triangle[0, 0:2]
    v2 = triangle[2, 0:2] - triangle[0, 0:2]
    x = np.array([x, y]) - triangle[0, 0:2]
    M = np.concatenate((v1, v2), axis=0).reshape(2, 2).transpose()
    sol = np.linalg.solve(M, x)
    if sol[0] > 0 and sol[1] > 0 and sol[0] + sol[1] < 1:
        return sol[0], sol[1]
    else:
        return False


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


def inside_rectangle(x, p, d, r, t_max):
    d_c = d.copy()
    d_c = d_c[0:2]
    d_c = d_c / np.sqrt(np.dot(d_c, d_c))

    t = np.dot(d_c, x[0:2] - p[0:2])
    g = np.dot(np.array([-d_c[1], d_c[0]]), x[0:2] - p[0:2])
    if 0 < t < t_max and -r < g < r:
        return True, t
    else:
        return False, t


def particle_mesh_intersection(p, d, r, vertices, faces, conexions, adjacency, h_max, h_min):
    # Gets the face where r(t) = p + t*d intersects with the mesh
    t_h = (h_max + r - p[2]) / d[2]
    p = p + t_h * d

    face = get_triangle(p[0], p[1], vertices, faces, adjacency)
    if not face: return False

    a = faces[face][0];
    b = faces[face][1];
    c = faces[face][2]
    ab = vertices[b][0:2] - vertices[a][0:2]
    bc = vertices[c][0:2] - vertices[b][0:2]
    ca = vertices[a][0:2] - vertices[c][0:2]
    t_ab = (ab[1] * p[0] + ab[0] * (vertices[a][1] - p[1]) - ab[1] * vertices[a][0]) / (
                d[1] * ab[0] - d[0] * ab[1] - 0.000001)
    t_bc = (bc[1] * p[0] + bc[0] * (vertices[b][1] - p[1]) - bc[1] * vertices[b][0]) / (
                d[1] * bc[0] - d[0] * bc[1] - 0.000001)
    t_ca = (ca[1] * p[0] + ca[0] * (vertices[c][1] - p[1]) - ca[1] * vertices[c][0]) / (
                d[1] * ca[0] - d[0] * ca[1] - 0.000001)

    # Looking for the base of the triangle
    if t_ab > 0:
        if t_bc < 0 and t_ca < 0:
            if t_bc < t_ca:
                b = c
            else:
                a = c
        else:
            if t_bc < t_ca:
                a = c
            else:
                b = c
    else:
        if 0 > t_bc > t_ab:
            a = c
        elif 0 > t_ca > t_ab:
            b = c

    visited = {face}
    v_explored = {a, b}
    v_to_explore = []
    intersection = False
    t_intersection = (h_min - p[2]) / d[2]

    while True:
        t, alpha, beta = Moller_Trumbore(vertices[faces[face]], p.copy(), d, False, r)
        #print(alpha, beta, end=" ")
        if alpha > 0 and beta > 0 and alpha + beta <= 1:
            t_intersection = t
            intersection = face
            break

        c = list(faces[face].copy())
        c.remove(a);
        c.remove(b);
        c = c[0]

        m = (vertices[a][0:2] + vertices[b][0:2]) / 2
        mc = vertices[c][0:2] - m
        ab = vertices[b][0:2] - vertices[a][0:2]
        perp = np.array([- mc[1], mc[0]])
        if np.dot(perp, ab) < 0: perp = np.array([mc[1], - mc[0]])

        direction = np.dot(perp, d[0:2])

        if direction == 0:
            gamma = (d[1] * p[0] + d[0] * (a[1] - p[1]) - d[1] * a[0]) / (d[1] * ab[0] - d[0] * ab[1])
            if gamma > 0.5:
                a = c
            elif gamma < 0.5:
                b = c
            else:
                print("Error c")
                return False
        else:
            gamma = (d[1] * p[0] + d[0] * (m[1] - p[1]) - d[1] * m[0]) / (d[1] * mc[0] - d[0] * mc[1])
            if gamma < 1 and direction < 0:
                b = c
            elif gamma < 1 and direction > 0:
                a = c
            elif gamma > 1 and direction < 0:
                a = c
            elif gamma > 1 and direction > 0:
                b = c
            else:
                print("Error c")
                return False

        comp, t_c = inside_rectangle(vertices[c], p, d, r, t_intersection)
        if comp: v_to_explore.append(c)
        if t_c > t_intersection: break

        for new_face in adjacency[face]:
            if a in faces[new_face] and b in faces[new_face] and new_face not in visited:
                #print(faces[face])
                face = new_face
                visited.add(new_face)
                break
        else:
            print("Leaves ", end="")
            break  # Leaves the mesh

    # print( v_to_explore )
    # print( intersection , t_intersection )

    while v_to_explore:
        v = v_to_explore.pop()
        v_explored.add(v)
        comp, t_v = inside_rectangle(vertices[v], p, d, r, t_intersection)
        if comp:
            for face in conexions[v]:
                if face not in visited:
                    visited.add(face)
                    t, alpha, beta = Moller_Trumbore(vertices[faces[face]], p.copy(), d, False, r)
                    # print( face , alpha , beta )
                    if alpha > 0 and beta > 0 and alpha + beta <= 1 and t < t_intersection:
                        t_intersection = t
                        intersection = face
                    for w in faces[face]:
                        if w not in v_explored:
                            v_to_explore.append(w)

    return intersection


def inverse_ray_mesh_interssection(p, d, face, vertices, faces, adjacency, h_lim):
    # Condition: p has to be inside the face vertices, at least in the XY plane
    a = faces[face][0];
    b = faces[face][1];
    c = faces[face][2]
    ab = vertices[b][0:2] - vertices[a][0:2]
    bc = vertices[c][0:2] - vertices[b][0:2]
    ca = vertices[a][0:2] - vertices[c][0:2]
    t_ab = (ab[1] * p[0] + ab[0] * (vertices[a][1] - p[1]) - ab[1] * vertices[a][0]) / (
                d[1] * ab[0] - d[0] * ab[1] - 0.000001)
    t_bc = (bc[1] * p[0] + bc[0] * (vertices[b][1] - p[1]) - bc[1] * vertices[b][0]) / (
                d[1] * bc[0] - d[0] * bc[1] - 0.000001)
    t_ca = (ca[1] * p[0] + ca[0] * (vertices[c][1] - p[1]) - ca[1] * vertices[c][0]) / (
                d[1] * ca[0] - d[0] * ca[1] - 0.000001)

    # Looking for the base of the triangle
    if t_ab > 0:
        if t_bc < 0 and t_ca < 0:
            if t_bc < t_ca:
                b = c
            else:
                a = c
        else:
            if t_bc < t_ca:
                a = c
            else:
                b = c
    else:
        if 0 > t_bc > t_ab:
            a = c
        elif 0 > t_ca > t_ab:
            b = c

    intersection = face
    visited = set([face])
    while True:
        t, alpha, beta = Moller_Trumbore(vertices[faces[face]], p.copy(), d)
        if alpha > 0 and beta > 0 and alpha + beta <= 1: intersection = face

        c = list(faces[face].copy())
        c.remove(a);
        c.remove(b);
        c = c[0]

        m = (vertices[a][0:2] + vertices[b][0:2]) / 2
        mc = vertices[c][0:2] - m
        ab = vertices[b][0:2] - vertices[a][0:2]

        t = (ab[1] * (vertices[a][0] - p[0]) + ab[0] * (p[1] - vertices[a][1])) / (d[0] * ab[1] - d[1] * ab[0])
        if p[2] + t * d[2] > h_lim: break

        perp = np.array([- mc[1], mc[0]])
        if np.dot(perp, ab) < 0: perp = np.array([mc[1], - mc[0]])

        direction = np.dot(perp, d[0:2])

        if direction == 0:
            gamma = (d[1] * p[0] + d[0] * (a[1] - p[1]) - d[1] * a[0]) / (d[1] * ab[0] - d[0] * ab[1])
            if gamma > 0.5:
                a = c
            elif gamma < 0.5:
                b = c
            else:
                break
        else:
            gamma = (d[1] * p[0] + d[0] * (m[1] - p[1]) - d[1] * m[0]) / (d[1] * mc[0] - d[0] * mc[1])
            if gamma < 1 and direction < 0:
                b = c
            elif gamma < 1 and direction > 0:
                a = c
            elif gamma > 1 and direction < 0:
                a = c
            elif gamma > 1 and direction > 0:
                b = c
            else:
                print("Error c")
                break

        for new_face in adjacency[face]:
            if a in faces[new_face] and b in faces[new_face] and new_face not in visited:
                face = new_face
                visited.add(new_face)
                break
        else:
            return False  # Leaves the mesh

    return intersection


def Monte_Carlo_Sample(phi, N, vertices, faces, adjacency, BB):
    directions_file = open("output_phi/directions_" + str(phi) + ".txt", "w")
    normals_file = open("output_phi/normals_" + str(phi) + ".txt", "w")
    starting_y, starting_f = starting_reference(vertices, conexions)
    normal, _ = get_normal(vertices, faces)

    h = BB[:, 1, 2].max()
    i = 0;
    while i < N:
        print(i)
        incidence = np.array([1411.22 * random(), 1057.72 * random(), h])
        start = np.array([0, 1057.72 * random(), 0])
        start[2] = np.sqrt(incidence[0] ** 2 + (incidence[1] - start[1]) ** 2) / np.tan(phi / 180 * np.pi) + h
        face = ray_mesh_intersection(start, incidence - start, vertices, faces, starting_y, starting_f, adjacency, BB)
        if face:
            i += 1
            directions_file.write(" ".join([str(d) for d in incidence - start]) + "\n")
            normals_file.write(" ".join([str(n) for n in normal[face]]) + "\n")
    directions_file.close()
    normals_file.close()


def mesh_characterization(phi, theta , r , a , b , vertices , faces , conexions , adjacency , h_max , h_min ):
    normals_file = open("output_part/normals_" + str(r) + "_" + str(phi) + "_" + str(theta) + ".txt", "a")
    normal, _ = get_normal(vertices, faces)
    if b > len(faces): b = len(faces)
    d = np.array([np.sin(phi / 180 * np.pi) * np.cos(theta / 180 * np.pi),
                  np.sin(phi / 180 * np.pi) * np.sin(theta / 180 * np.pi), np.cos(phi / 180 * np.pi)])
    for face in range(a, b):
        incidence = sum(vertices[faces[face]]) / 3
        face_i = particle_mesh_intersection(incidence, d, r , vertices, faces, conexions , adjacency, h_max , h_min )
        print( face , face_i )
        if face_i: normals_file.write(str(face_i) + "\n")
    normals_file.close()


if __name__ == "__main__":
    vertices, faces, conexions = read_mesh("input/mucus_large.txt")
    adjacency = adjacency_graph( conexions , faces )
    # BB = bounding_box( vertices , faces )
    #adjacency, BB = read_extra_data("large")

    # h_lim = max(BB[:,1,2])
    # for theta in [  270 , 180 , 45 , 90 , 135, 225 , 315  ]:
    #    for phi in [ 60 ]:
    #        mesh_characterization( phi , theta , 0 , 4000000 , vertices , faces , adjacency , h_lim )

    #mesh_characterization(60, 0, 50 ,0 , 4000000, vertices, faces , conexions , adjacency, max(BB[:, 1, 2]) , min(BB[:, 0, 2]))

    # starting_y , starting_f = starting_reference( vertices , conexions )
    # normal, faces_area = get_normal(vertices, faces)
    # phi, theta = spheric_angles(normal.copy())
    # bounces = bouncings(normal, np.array([1,0,-1]))
    # bphi, btheta = spheric_angles(bounces.copy())



##################################


def __p():
    plt.plot(phi, theta, "ko", markersize=2)
    plt.xlabel("phi")
    plt.ylabel("theta")
    plt.grid()
    plt.show()

    print(pearsonr(phi, theta))
    print(np.cov(phi, theta))

    pd.DataFrame(phi).plot(kind='density', color="black", xlim=(0, 90))
    plt.grid()
    plt.xlabel("phi")
    plt.legend(["density"])
    plt.show()

    pd.DataFrame(theta).plot(kind='density', color="black", xlim=(-190, 190))
    plt.plot(np.linspace(-180, 180, 200), np.ones(200) / 360, color="red")
    plt.grid()
    plt.xlabel("theta")
    plt.legend(["density", "uniform distribution"])
    plt.show()

    plt.plot(bphi, btheta, "ko", markersize=2)
    plt.xlabel("phi")
    plt.ylabel("theta")
    plt.grid()
    plt.show()


def __angle_print():
    print(pearsonr(phi, theta))
    print(np.cov(phi, theta))

    pd.DataFrame(bphi).plot(kind='density', color="black", xlim=(0, 180))
    plt.grid()
    plt.xlabel("phi")
    plt.legend(["density"])
    plt.show()

    pd.DataFrame(btheta).plot(kind='density', color="black", xlim=(-190, 190))
    plt.grid()
    plt.xlabel("theta")
    plt.legend(["density"])
    plt.show()
