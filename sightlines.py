import numpy as np
import pandas as pd

# the code returns the density array 'den', and vz array 'vz'
# den has shape[N_sight,N_sight,nbins], where den[i,j] will give the density profile of a sightline which is ith along x and jth along y
# vz has the same shape, and gives the mean vz velocity (number weighted)
# one should provide the arrays Pos and Velz, in the same shape as metioned below.
#######################################################################################
# parameters
l=150 # box size in Mpc/h
N_dm= 1024**3 #number of particles
m_dm= 0.192605/8 #mass of particles in M_sun/h
N_sight= 70 # number of sightlines per side
D=150/1024 #diameter of the sightlines
nbins= 70 # number of bins along the los


# replace Pos and Vel by the actual data
Pos=np.zeros([1024**3,3])
Velz=np.zeros(1024**3)

############################################################################
nbox=70*3 #number of sub-boxes the simulation box is divided into
lbox=l/nbox # length of the sub-boxes

dx=l/N_sight # distance between sightlines
x_sight=np.arange( dx/2, l, dx) # positions of sightlines
#######################################################
def W(a,b,c, nbox):
    ''' gives the weight of the box given its x y and z indices and the number of cells along an axis'''
    a=a%nbox #periodic boundary conditions
    b=b%nbox
    c=c%nbox
    return (a*nbox**2+b*nbox+c)

def split(l,Pos, nbox):
    """ This function gives the indices of the particles such that they are arranged into small sub-boxes"""
    lbox=l/nbox
    edge=np.array(np.linspace(lbox, l, nbox)) #left edge of the sub-boxes
    rind=np.array(Pos/lbox, dtype=int) #(i,j,k) index of the box that the particle belongs to
    weight=rind[:,0]*nbox**2+rind[:,1]*nbox+rind[:,2] # index of the box, when the 3D array is flattened
    rind=0

    sort=weight.argsort()  #finally, one has to do Pos[sort], Vel[sort]
    weight=weight[sort]
    Num_W=np.bincount(weight)  #number of particles in each sub-box
    weight=0
    Num_cum=np.cumsum(Num_W) #number of particles below the given sub-box
    #make sure that there are non-zero particles per sub-box, otherwise Num-cum has dimensions different from number of sub-boxes, and the code will fial.
    Num_cum=np.roll(Num_cum, 1)
    Num_cum[0]=0

    return sort, Num_cum, Num_W 
#############################################################

sort, Num_cum, Num_W=split(l,Pos, nbox)
Pos=Pos[sort]
Velz=Velz[sort]

zbins=np.linspace(0, l, nbins+1)
den=np.zeros([N_sight, N_sight, nbins],dtype=float)
vz=np.zeros_like(den)


for i in range(len(x_sight)):
    for j in range(len(x_sight)):
        x0=x_sight[i]
        y0=x_sight[j]
        
        minr=np.array([x0-D/2, y0-D/2, 0])
        maxr=np.array([x0+D/2, y0+D/2, l])
  
        indmax=np.array(np.floor(maxr/lbox), dtype=int) #maximum index of sub-boxes along each direction
        indmin=np.array(np.floor(minr/lbox), dtype=int)


        xind=np.array(np.arange(indmin[0], indmax[0]+1,1), dtype=int)
        yind=np.array(np.arange(indmin[1], indmax[1]+1,1), dtype=int)
        zind=np.array(np.arange(indmin[2], indmax[2]+1,1), dtype=int)
        X, Y, Z= np.meshgrid(xind, yind, zind)
        X=X.flatten()
        Y=Y.flatten()
        Z=Z.flatten()
        weights=W(X, Y, Z, nbox) #sub-boxes to search in this takes into account PBC
        weights=np.unique(weights)
        
        X=Y=Z=0 #clear space

        weight0=weights[0] #first sub-box
        
        Pos2=Pos[Num_cum[weight0]:Num_cum[weight0]+Num_W[weight0],:]
        Velz2=Velz[Num_cum[weight0]:Num_cum[weight0]+Num_W[weight0]]
        
        
        for b in range(1,len(weights),1):
            Pos1=Pos[Num_cum[weights[b]]:Num_cum[weights[b]]+Num_W[weights[b]],:]
            Pos2=np.concatenate([Pos2,Pos1],axis=0)
            
            Velz1=Velz[Num_cum[weights[b]]:Num_cum[weights[b]]+Num_W[weights[b]]]
            Velz2=np.concatenate([Velz2,Velz1],axis=0)
            
        r_perp=np.sqrt((Pos2[:,0]-x0)**2+(Pos2[:,1]-y0)**2)
        ind1=np.where(np.abs(r_perp)<D/2)[0]
        
        Pos2=Pos2[ind1,:]
        Velz2=Velz2[ind1]
        for k in range(nbins):
            ind=np.where((Pos2[:,2]>=zbins[k])&(Pos2[:,2]<zbins[k+1]))[0]
            
            den[i][j][k]=len(ind)
            vz[i][j][k]=np.mean(Velz2[ind])

vz[den==0]=0 # when there are no particles, set vz=0

vol=np.pi*(D/2)**2*(zbins[1]-zbins[0]) # volume of each sub-tube
den=den*m_dm/vol # density in units of 10^10 solar mass/h /(Mpc/h)^3

            
            

