import numpy as np

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


def read_part( r , phi , theta , faces , conexions , norm ):
    #normals_file = open( "output_part/normals_" + str(r) + "_" + str(phi) + "_" + str(theta) + ".txt" , "r" )
    normals = []
    for line in open( "output_part/normals_" + str(r) + "_" + str(phi) + "_" + str(theta) + ".txt" , "r" ):
        line = line.split()
        if line[0] == "0": 
            faces = np.array( list( conexions[int(line[1])] ) )
            faces = norm[ faces ]
            n = sum( faces )/len(faces)
            normals.append( n ) 
        elif line[0] == "1":
            face = int(line[1])
            normals.append( norm[face] )
        elif line[0] == "2":
            faces = np.array(  list( conexions[int(line[1])].intersection( conexions[int(line[2])] ) ) )
            faces = norm[ faces ]
            n = sum( faces )/len(faces)
            normals.append( n )             

    return normals


def write_spheric( R , PHI , THETA , faces , conexions , normal ):
    for r in R:
        for phi in PHI:
            for theta in THETA:
                print( r , phi , theta )
                n = read_part( r , phi , theta , faces , conexions , normal )
                dir = np.array( [ -np.sin(phi/180*np.pi)*np.cos(theta/180*np.pi), np.sin(phi/180*np.pi)*np.sin(theta/180*np.pi) , -np.cos(phi/180*np.pi)])
                nphi, ntheta = spheric_angles_dir( n.copy() , dir)
                spheric = open( "output_part/normals/spheric_" + str(r) + "_" + str(phi) + "_" + str(theta) + ".txt" , "w" )
                for i in range(len(nphi)):
                    spheric.write( str(nphi[i]) + " " + str(ntheta[i]) + "\n" ) 

def spheric_angles_dir( normal , d ):
    #Converts
    x = np.array([1, 0, 0]); y = np.array([0,1,0]); z = np.array([0,0,1])
    phi = []; theta = []
    for i,n in enumerate(normal):
        n = n / np.sqrt(np.dot(n, n))
        x = np.array([ d[0] , d[1] , 0 ])
        x = x / np.sqrt(np.dot(x, x))
        y = np.array([ x[1] , -x[0] , 0 ])
        phi.append(np.arccos(np.dot( z , n ) )/np.pi*180)
        n[2] = 0; n = n / np.sqrt(np.dot(n,n))
        vx = np.dot( x , n ); vy = np.dot( y , n )
        if vy >= 0:
            theta.append( np.arccos(vx)/np.pi*180)
        else:
            theta.append( -np.arccos(vx)/ np.pi * 180)
    return np.array(phi), np.array(theta)


print("ok")
vertices, faces, conexions , edges = read_mesh("input/mucus_large.txt")
print("ok")
normal, faces_area = get_normal(vertices, faces)
print("ok")
write_spheric( [5,10,25,50] , [60],[0] , faces , conexions , normal )



















