import numpy as np
from time import perf_counter as timer
from numba import njit , objmode , prange

def read_mesh(filename):
    #vertices: np array with the coordinates of each vertex
    #faces: np array with the three vertices that forms the face
    #conexions: set of faces around each vertex
    #edegs: set of vertex around each vertex

    mesh = open(filename)
    n, m = mesh.readline().split()
    n = int(n)
    m = int(m)
    vertices = []
    faces = []
    conexions = [set() for i in range(n)]
    edges = [set() for i in range(n)]

    for i in range(n):
        vertices.append(mesh.readline().split())
        for j in range(3):
            vertices[i][j] = float(vertices[i][j])

    for i in range(m):
        faces.append(mesh.readline().split())
        for j in range(3):
            faces[i][j] = int(faces[i][j]) - 1
            conexions[faces[i][j]].add(i)
            for k in range(j-1,-1,-1):
                edges[faces[i][j]].add(faces[i][k])
                edges[faces[i][k]].add(faces[i][j])

    return np.array(vertices), np.array(faces) , conexions , edges

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

@njit(parallel = True)
def cylinder_mesh_intersection( p , d , r , vertices ):
    #with objmode( at='f8'): at = timer()
    indexes = np.arange(len(vertices))
    #with objmode( bt='f8'):bt = timer()
    #d = d/np.sqrt(np.dot(d,d))
    x = np.subtract(vertices,p)
    #with objmode( ct='f8'):ct = timer()
    b = np.dot( x , d )
    #with objmode( dt='f8'):dt = timer()
    t2 = r**2 - np.sum((x*x),axis = 1) + b*b
    #with objmode( et='f8'):et = timer()
    indexes = indexes[ t2 > 0 ]
    #with objmode(ft='f8'): ft = timer()
    #print( bt-at , ct-bt , dt - ct , et - dt, ft - et )
    return indexes, b[indexes]-np.sqrt(t2[indexes])

def improved_cylinder_mesh_intersection( p , d , r , vertices  , faces , adjacency , edges , h_0):
    #d = d / np.sqrt(np.dot(d, d))

    t_0 = (h_0-p[2])/d[2]
    x_0 = p[0]+t_0*d[0]
    y_0 = p[1]+t_0*d[1]
    face_0 = get_triangle( x_0 , y_0 , vertices , faces , adjacency )
    r2 = r*r

    to_visit = list(faces[face_0])
    visited = set(to_visit)
    indexes = []
    t_indexes = []

    while len(to_visit)>0:
        v = to_visit.pop()
        d_cil = r2 + np.dot( d , p - vertices[v])**2 - np.dot( p-vertices[v] , p-vertices[v] )
        if d_cil > -50 :
            to_visit.extend( list(edges[v].difference(visited)) )
            visited.update( edges[v] )
        if d_cil > 0 :
            indexes.append( v )
            t_indexes.append( -np.dot( d , p - vertices[v]) - np.sqrt(d_cil) )

    return np.array(indexes) , np.array(t_indexes)

#@njit(parallel=True)
def sphere_line_intersection( p , d , r , a , b ):
    #intersection between p+td , a(1-g)+bg
    v = b - a
    n = np.cross( v , d )
    n = n/np.sqrt(np.dot(n,n))
    e = np.dot(p-a,n)
    if e**2 > r**2: return False
    r1=np.sqrt(r**2-e**2)
    p1 = p-e*n
    s = np.cross(v,n)
    if np.dot(s,s) == 0:return 1000
    s = s/np.sqrt(np.dot(s,s))
    if np.dot(s,d) < 0: s = -s
    p2 = p1+r1*s

    if (v[0]*d[1]-v[1]*d[0])==0 : return 1000
    t = (v[1]*(p2[0]-a[0]) - v[0]*(p2[1]-a[1]))/(v[0]*d[1]-v[1]*d[0])
    if v[0]!= 0:
        g = (p2[0]-a[0]+t*d[0])/v[0]
    elif v[1] != 0:
        g = (p2[1] - a[1] + t * d[1]) / v[1]
    elif v[2] != 0:
        g = (p2[2] - a[2] + t * d[2]) / v[2]

    if g<0 or g>1 : return False
    return (v[1]*(p2[0]-a[0]) - v[0]*(p2[1]-a[1]))/(v[0]*d[1]-v[1]*d[0])

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

#@njit(parallel = True )
def Moller_Trumbore(triangle, p, d, n=False, r=0):
    # gets the coordinates of the intersection of r(t) = p+t*d and the triangle, T(alpha,beta) = alpha*v1 + beta*v2
    v1 = triangle[1] - triangle[0]
    v2 = triangle[2] - triangle[0]
    if not n:  # computes normal vector
        n1 = np.cross(v1, v2)
        n2 = n1 / np.sqrt(np.dot(n1, n1))

    p = p - n2 * r

    M = np.concatenate((-d, v1, v2)).reshape(3, 3).transpose()
    sol = np.linalg.solve(M, p - triangle[0])
    return sol[0], sol[1], sol[2]

def particle_mesh_intersection( p , d , r , vertices , faces , conexions , edges , h_max ):
    #d = d/np.sqrt(np.dot(d, d))
    #index_v , t_v = improved_cylinder_mesh_intersection( p , d , r , vertices  , faces , adjacency , edges , (h_max+h_min)/2 )
    a = timer()
    index_v , t_v = cylinder_mesh_intersection( p , d , r , vertices )
    print( timer() - a )
    if len(index_v) == 0: return False
    kind = 0
    t_min = min(t_v)
    p_min = index_v[np.where(t_v == t_min)[0][0]]
    index_v = index_v[t_v < t_min + 1]
    #t_v = t_v[t_v < t_min + 1]
    #count = 0
    for i in range( len(index_v) ):
        v = index_v[i]
        for face in conexions[v]:
            t , alpha , beta = Moller_Trumbore( vertices[faces[face]] , p , d , False , r )
            if t < t_min and alpha > 0 and beta > 0 and alpha + beta < 1:
                kind = 1
                t_min = t
                p_min = face
                #count += 1
        for edge in edges[v]:
            t = sphere_line_intersection( p , d , r , vertices[v] , vertices[edge] )
            if t and t < t_min:
                kind = 2
                t_min = t
                p_min = [v,edge]
                #count += 1

    margin = np.sqrt(r**2-(p[2]+t_min*d[2]-h_max)**2)
    if p[0] + t_min * d[0] < margin or p[0] + t_min * d[0] > 1411 - margin or p[1] + t_min * d[1] < margin or p[1] + t_min * d[1] > 1057 - margin: return False

    if kind == 2:
        return kind , p_min[0] , p_min[1] , t_min

    return kind , p_min , t_min

def mesh_characterization( phi, theta , r , a , b , vertices , faces , conexions , edges , h_max , h_min ):
    #normals_file = open("output_part/normals_" + str(r) + "_" + str(phi) + "_" + str(theta) + ".txt", "a")
    normal, _ = get_normal(vertices, faces)
    if b > len(faces): b = len(faces)
    d = np.array([np.sin(phi / 180 * np.pi) * np.cos(theta / 180 * np.pi),
                  np.sin(phi / 180 * np.pi) * np.sin(theta / 180 * np.pi),
                  -np.cos(phi / 180 * np.pi)])
    for face in range(a, b):
        incidence = sum(vertices[faces[face]]) / 3
        face_i = particle_mesh_intersection(incidence, d , r , vertices, faces, conexions , edges , h_max )
        print( face , face_i )
        # if face_i:
            #normals_file.write(" ".join([str(adj) for adj in face_i]) + "\n" )
    #normals_file.close()


if __name__ == "__main__":
    vertices , faces , conexions, edges = read_mesh("input/mucus_large.txt")
    adjacency, BB = read_extra_data("large")

    for phi in [60]:
        for r in [25]:
            mesh_characterization(phi , 0 , r , 0 , 100000 , vertices , faces , conexions , edges , max(BB[:, 1, 2]) ,min(BB[:, 0,     2]))
    #mesh_characterization(60, 0, 50, 80000, 100000, vertices, faces, conexions, edges, max(BB[:, 1, 2]) , min(BB[:, 0, 2]))



















