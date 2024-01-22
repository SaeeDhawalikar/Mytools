import numpy as np
from scipy.spatial import Delaunay

class Dela(object):
    ''' Calculates the Delaunay Tessellation "Tess" assuming the given boundary conditions
    for points with catesian coordinates
    Pos: [N, D] array, cartesian coordinates of the points where
        D: dimension of space
        N: number of points
    PBC: defalut True'''
    def __init__(self, Pos,L=None, PBC=True):
        self.D=len(Pos[0]) # spatial dimension
        self.N=len(Pos[:,0])
        
        if (PBC is False):
            self.Tess=Delaunay(Pos)
            POS=Pos.copy()
            
        else:
            if (L is None):
                self.L=np.max(Pos)+1e-3
            else:
                self.L=L
            # creating a periodic cube     
            add=np.array([-self.L, 0, self.L])
            POS=Pos.copy()

            if (self.D==2):
                for i in range(len(add)):
                    for j in range(len(add)):
                        if((i==1)&(j==1)):
                            continue
                        pos_new=Pos+np.array([add[i], add[j]])
                        POS=np.concatenate([POS, pos_new], axis=0)
                self.Tess=Delaunay(POS)
            elif (self.D==3):
                for i in range(len(add)):
                    for j in range(len(add)):
                        for h in range(len(add)):
                            if((i==1)&(j==1)&(h==1)):
                                continue
                        pos_new=Pos+np.array([add[i], add[j], add[h]])
                        POS=np.concatenate([POS, pos_new], axis=0)
                self.Tess=Delaunay(POS)
        self.POS=POS
        self.Pos=Pos
    def find_Delaneigh(self, itself=True, index=None):
        '''returns the delaunay neighbour vertices of all the points. 
        Inputs:
        itself: defalut True. Includes the point itself along with its neighbours
        index: index of the point for which to find the neighbours.
            by default, it is None, and neighbours of all the points are returned
        Outputs:
        Nei: list of N arrays, where kth array gives the indices of its Delaunay neighbours
            if index is given, it is a single array.'''
        if index is None:
            Nei=[]
            pts, nei=self.Tess.vertex_neighbor_vertices
            for k in range(self.N):
                nei_k=nei[pts[k]:pts[k+1]]
                nei_k=nei_k%N  # accounting for PBC
                if (itself is True):
                    nei_k=np.append(nei_k, k)
                Nei.append(nei_k)
        else:
            index=int(index)
            Nei=Nei[index]
        return Nei
    
    
    def Volume(self,A, POS=None):
        ''' returns the volume of a D-dimensional tetrahedron with vertices A
        Here,
        Pos:positions of points. No need to specify this.
        A is an array of indices of the vertices.
        reutrns:
        V: volume of the cell'''
        if POS is None:
            POS=self.POS
        if (self.D==3):
            temp1=POS[A[0]]-POS[A[3]]
            temp2=POS[A[1]]-POS[A[3]]
            temp3=POS[A[2]]-POS[A[3]]
            
            temp4=np.cross(temp2, temp3)
            V=np.dot(temp1,temp4)
            V=np.abs(V)/6
        elif (self.D==2):
            x=POS[A,0]
            y=POS[A,1]
            V=x[0]*(y[1]-y[2])+x[1]*(y[2]-y[0])+x[2]*(y[0]-y[1])
            V=np.abs(V)/2
        return V
    
    def DTFE(self):
        """ returns the density at every point estimated usin Delaunay Tessellation Field Estimator.
        output: DTFE (array of length N)"""
        
        V_to_S=self.Tess.vertex_to_simplex # simplex associated with each vertex
        Neigh=self.Tess.neighbors # neighbouring simplices of each simplex
        S=self.Tess.simplices # vertices of simplices arranged in a 2d array

        Vol=np.zeros(self.N)# volumes 
        for k in range(self.N):
            Sim1=V_to_S[k] #index of the simplex
            Vert1=S[Sim1] # indices of the vertices of the simplex Sim1
            Vol[k]+=self.Volume(Vert1)
            taken=[Sim1] # simplices whose volume is already added

            # looping over the adjacent simplices
            Nei1=Neigh[Sim1]
            i=0
            while (i <len(Nei1)):
                S_curr=Nei1[i] # current simplex
                V_curr=S[S_curr] # vertices of the current simplex
                if ((k in V_curr%self.N)&(S_curr not in taken)): # if the simplex contains the point of interest or its shifted copy
                    Vol[k]+=self.Volume(V_curr) #note volume is calculated without PBC
                    Nei1=np.append(Nei1, Neigh[S_curr])
                    taken.append(S_curr)
                i=i+1
        # mass per particle=1/N, so that the total mass is 1
        m=1/(self.N)
        DTFE=m*(1+self.D)/Vol
        return DTFE
    
        
        
        
        
            
        
        
 
