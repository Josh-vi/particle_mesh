import numpy as np
from matplotlib import pyplot as plt

class distribution:
    def __init__(self,radious,partitions):
        self.r = np.array( radious , dtype = float  )
        self.p = np.array( partitions , dtype = int )
        if self.p[0] != 1: raise Exception( "First element in partitions must be 1" )
        self.n = len(self.r)
        if self.n != len(partitions): raise Exception( "Error, lists of different lengths" )
        self.blocks = np.sum(self.p)
        self.block_size = self.blocks_size()
        self.cumulative_p()
        self.main_points()
        self.color_scale = 300

    def __str__(self): #output a file
        show = ""
        show += " ".join( [str(r) for r in self.r ])+"\n"
        show += " ".join([str(p) for p in self.p]) + "\n \n"
        prob = 0
        for i in range(self.blocks):
            show += "{:.1f}".format(self.phi[i])+" "+"{:.1f}".format(self.theta[i])
            try:
                prob += self.hystogram[i]
                show += " "+str(self.hystogram[i])+" "+str(prob)+"\n"
            except: show += "\n"
        return show

    def blocks_size(self):
        block_size = [np.pi*self.r[0]**2]
        for layer in range(1,self.n):
            size = (self.r[layer]**2 - self.r[layer-1]**2)*np.pi/self.p[layer]
            for b in range(self.p[layer]):
                block_size.append( size )
        return block_size

    def cumulative_p(self):
        self.cum_p = [0]
        for i in range(len(self.p) ):
            self.cum_p.append( self.cum_p[-1] + self.p[i] )

    def main_points(self):
        self.phi = []
        self.theta = []
        for layer in range(self.n):
            for block in range(self.p[layer]):
                if layer == 0: self.phi.append( 0 )
                else: self.phi.append( (self.r[layer]+self.r[layer-1])/2 )

                theta = 360./self.p[layer]*block
                if theta > 180 : theta = - 360 + theta
                self.theta.append(theta)

        self.phi = np.array(self.phi)
        self.theta = np.array(self.theta)

    def get_block(self,phi,theta):
        if phi > self.r[-1] or phi < 0: return -1
        block_size = 360./self.p[np.sum( self.r < phi )]
        theta = theta + block_size / 2
        if theta < 0 : theta = 360 + theta
        block_idx = np.floor( theta / block_size ).astype(int)
        return np.sum( self.p[self.r<phi] ) + block_idx

    def build_hystogram(self,file):
        self.hystogram = np.zeros(self.blocks)
        for line in open( file , "r" ):
            line = line.split()
            phi , theta = [float(l) for l in line ]
            block = self.get_block(phi,theta)
            if block>= 0: self.hystogram[block] += 1
        self.hystogram = self.hystogram / sum(self.hystogram)

    def iglu_plot(self):
        plt.figure(figsize=(10, 10))
        scale = 15
        cmap = plt.colormaps["jet"]
        map = self.hystogram / self.block_size
        map = map * self.color_scale
        for layer in range(len(self.r)-1,-1,-1):
            #theta = np.arange(self.p[layer])*360./self.p[layer]
            plt.pie(  np.ones(self.p[layer])/self.p[layer]  ,
                     colors = cmap( map[self.cum_p[layer]:self.cum_p[layer+1]] ),
                     radius = self.r[layer]/scale , startangle = -90-180./self.p[layer],
                     wedgeprops = {'linewidth' : 2 , 'edgecolor' : 'black'})
        x = self.phi*np.cos((self.theta-90)/180*np.pi)/scale
        y = self.phi*np.sin((self.theta-90)/180*np.pi)/scale
        plt.scatter( x , y , color = "black"  )
        plt.xlim(-1.8,1.8)
        plt.ylim(-1.8,1.8)
        plt.show()

    def brick_wall_plot(self):
        plt.figure(figsize=(10,10))
        scale = 15
        cmap = plt.colormaps["jet"]
        X =  [0] + list(self.r)
        Y = np.linspace( -180 , 180 , 2*np.lcm.reduce(self.p) )
        Z = np.arange((len(X)-1)*(len(Y)-1),dtype = float).reshape((len(X)-1,len(Y)-1))

        for i in range(len(X)-1):
            for j in range(len(Y)-1):
                x = (X[i] + X[i+1])/2
                y = (Y[j] + Y[j+1])/2
                block = self.get_block(x,y)
                Z[i,j] = self.hystogram[block] /self.block_size[block] * self.color_scale

        plt.pcolormesh(X , Y , Z.transpose() , shading = "flat" , cmap = "jet" , vmin = 0 , vmax = 1 )
        #plt.scatter( self.phi , self.theta , color = "black"  )
        #plt.scatter(self.phi[self.theta ==180], -self.theta[self.theta ==180], color="black")
        plt.show()


class distribution_cartessian:
    def __init__(self,X):
        X = [ -X[i] for i in range(len(X)-1,-1,-1) ] + X
        self.X_cuts = np.array(X)
        self.Y_cuts = np.array(X)
        self.X = ( self.X_cuts[:-1] + self.X_cuts[1:] )/2
        self.Y = ( self.Y_cuts[:-1] + self.Y_cuts[1:] )/2
        self.nX = len( self.X )
        self.nY = len( self.Y )

        self.blocks , self.block_size = self.get_blocks()
        self.color_scale = 300

    def __str__(self):
        show = ""
        show += " ".join( [str(x) for x in self.X_cuts ])+"\n"
        show += " ".join([str(y) for y in self.Y_cuts]) + "\n \n"
        prob = 0
        for i,(x,y) in enumerate(self.blocks):
            show += str(x)+"\t"+str(y)
            try:
                prob += self.hystogram[i]
                show += "\t"+str(self.hystogram[i])+"\t"+str(prob)+"\n"
            except: show += "\n"
        return show

    def get_blocks(self):
        blocks = []
        size =[]
        for j in range(self.nY):
            y = self.Y[j]
            for i in range(self.nX):
                x = self.X[i]
                blocks.append( [x,y] )
                size.append( (self.X_cuts[i+1]-self.X_cuts[i])*(self.Y_cuts[j+1]-self.Y_cuts[j]) )
        return np.array(blocks) , np.array(size)


    def get_block(self,x,y):
        if x < self.X_cuts[0] or x > self.X_cuts[-1] or y < self.Y_cuts[0] or y > self.Y_cuts[-1] : return False
        return (np.sum( self.X_cuts < x )-1)+(np.sum( self.Y_cuts < y )-1)*self.nX

    def build_hystogram(self,file):
        self.hystogram = np.zeros(self.nX*self.nY)
        for line in open( file , "r" ):
            line = line.split()
            phi, theta = [float(l) for l in line]
            x = phi*np.cos(theta/180*np.pi)
            y = phi*np.sin(theta/180*np.pi)
            block = self.get_block(x,y)
            if block: self.hystogram[block] += 1
        self.hystogram = self.hystogram / sum(self.hystogram)

    def brick_wall_plot(self , curves = False ):
        plt.figure(figsize=(20,20))
        Z = self.hystogram/self.block_size*self.color_scale
        Z = Z.reshape( (self.nX,self.nY) )

        plt.pcolormesh(self.X_cuts , -self.Y_cuts , Z.transpose() , shading = "flat" , cmap = "jet" , vmin = 0 , vmax = 1 ,
                       edgecolor = "black" , linewidth = 1)
        plt.scatter( self.blocks[:,0] , self.blocks[:,1] , color = "black"  )

        if curves:
            theta = np.linspace( -np.pi,np.pi , 100 )
            for c in curves:
                x = np.cos(theta)*c
                y = np.sin(theta)*c
                plt.plot( x , y , "r--" , linewidth = 4 )

        plt.show()

def read_distribution(filename):
    hyst = open(filename)
    radious = hyst.readline().split()
    radious = [int(r) for r in radious ]
    partitions = hyst.readline().split()
    partitions = [int(p) for p in partitions ]
    blocks = sum( partitions )
    h = distribution( radious , partitions )
    h.hystogram = []
    for b in range(blocks):
        phi , theta , prob , dens = hyst.readline().split()
        prob = float(prob)
        h.hystogram.append( prob )
    h.hystogram = np.array(h.hystogram)

    return h


if __name__ == "__main__": #Set tasks input
    r = 10
    phi = 10
    a = read_distribution("hyst/hystogram_" + str(r) + "_" + str(phi) + ".txt")
    a.color_scale = 400
    a.iglu_plot()
    print( a )


