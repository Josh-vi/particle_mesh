import numpy as np
import matplotlib.pylab as plt
from scipy.stats import pearsonr
import pandas as pd
import seaborn as nsn

def read_data( phi , N ):
    directions_file = open("output_phi/directions_" + str(phi) + ".txt", "r")
    normals_file = open("output_phi/normals_" + str(phi) + ".txt", "r")
    directions = []
    normals = []
    for i in range(N):
        directions.append(directions_file.readline().split())
        normals.append(normals_file.readline().split())
        for j in range(3):
            directions[i][j] = float(directions[i][j])
            normals[i][j] = float(normals[i][j])
    directions_file.close()
    normals_file.close()
    return np.array(directions) , np.array(normals)

def spheric_angles( normal , dircetions ):
    #Converts
    x = np.array([1, 0, 0]); y = np.array([0,1,0]); z = np.array([0,0,1])
    phi = []; theta = []
    for i,n in enumerate(normal):
        d = dircetions[i]
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

def bouncings( normal , direction ):
    bounces = []
    for i in range( len(normal) ):
        n = normal[i]
        d = direction[i]
        d = d / np.sqrt(np.dot(d, d))
        bounces.append(bounce(n,d))
    return np.array(bounces)

def bounce( n , x ):
    # return elastic colission of x in n
    R = 2*np.kron( n , n ).reshape((3,3))-np.identity(3)
    return np.matmul( R , -x )

def plot_normals( p , phi , theta ):
    plt.plot(phi, theta, "ko", markersize=2)
    plt.title("Incidence phi = " + str(p))
    plt.xlabel("phi")
    plt.ylabel("theta")
    plt.grid()
    plt.show()

    pd.DataFrame(phi).plot(kind='density', color="black", xlim=(0, 180))
    plt.grid()
    plt.title("Incidence phi = " + str(p))
    plt.xlabel("phi")
    plt.legend(["density"])
    plt.show()

    pd.DataFrame(theta).plot(kind='density', color="black", xlim=(-190, 190))
    plt.plot(np.linspace(-180, 180, 200), np.ones(200) / 360, color="red")
    plt.grid()
    plt.title("Incidence phi = " + str(p))
    plt.xlabel("theta")
    plt.legend(["density", "uniform distribution"])
    plt.show()

if __name__ == "__main__":
    p = 65
    directions , normals = read_data( p , 1000 )
    phi, theta = spheric_angles(normals.copy() , directions.copy() )
    bounces = bouncings(normals.copy(), directions.copy() )
    bphi, btheta = spheric_angles(bounces.copy() , directions.copy())

    #plot_normals( p , phi , theta )
    plot_normals( p , bphi , btheta )


