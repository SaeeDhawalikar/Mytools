import numpy as np
import pandas as pd
import h5py
import re
class Disperse:
    def __init__(self, l=300):
        self.l=l
    
    def bound_cond(self,x, l):
        return x%l
    
    def delaunay(self, file, D=3):
        ''' outputs: ar, df
        ar: [N, D+2] array, where N is the number of tracers
            first D columns contain the positions of tracers (vertices) last 2 columns the corresponding density and log density
        df is a dataframe containing the following information
        N: number of tracers (vertices)
        n_face: number of D-simplexes
        n_face1: number of (D-1) simplexes if D=3
        n_edge: number of edges
        
        inputs:
        data: .ply file
        D: dimension of space'''

        a="end_header"
        ver="element vertex"
        fa="element face" # D-simplex (tetrahedron in 3d, triangle in 2d)
        fa1="element 2-face" #triangle in 3d, not present in 3d
        ed="element 1-face" #edge
        
        n_face=0
        n_face1=0
        n_edge=0
        N=0

        start=0
        data=pd.read_csv(file,  header=None)
        for i in range(len(data)):
            line=str(np.array(data[i:i+1]))
            if ver in line:
                N=np.array(re.findall('\d+', line))
                N=int(N[-1])

            if fa in line:
                n_face=np.array(re.findall('\d+', line))
                n_face=int(n_face[-1])
                
            if fa1 in line:
                n_face1=np.array(re.findall('\d+', line))
                n_face1=int(n_face1[-1])

            if ed in line:
                n_edge=np.array(re.findall('\d+', line))
                n_edge=int(n_edge[-1])

            if a in line:
                start=i+1
                break


        data_dela=data[start:start+N]
        vertex=list(data_dela.to_numpy())

        ar=[]
        for i in range(len(vertex)):
            hi=vertex[i][0].split(' ')
            temp=np.array(hi[0:D+2], dtype=float)
            ar.append(temp)
        ar=np.array(ar, dtype=float)
        
        df=pd.DataFrame({"N":[N], "n_face":[n_face], "n_face1":[n_face1], "n_edge":[n_edge]})

        return ar, df
    
    def find_fil(self,data):
        ''' Gives the starting point to read filament data, and filament number
        input:
        data: filament file data in pandas DataFrame'''
        start=0
        a="[FILAMENTS]"
        for i in range(len(data)):
            line=np.array(data[i:i+1])
            if a in str(line):
                start=i+2
        fil_num=int(np.array(data[start-1:start]))
        return (start, fil_num)
    
    def plot_fil(self,data, start, fil_num):
        '''Gives positions of points in the filament, split into different arrays so that proper plotting is possible
        INPUTS:
        data: pandas DataFrame of the filament file
        start: starting point of the filaments, given by the function find_fil
        fil_num: number of filaments
        
        OUTPUT:
        out: a list in the form [[r1, r2, r3, r3]_i], where i is the index of the filament, and points are split into 
        different componets so that periodic boundary conditions don't give weird looking filaments'''
        out=[]
        l=self.l

        new=np.array(data[start:])
        j=0
        for i in range(fil_num):
            fil_head=np.array(new[j][0].split(), dtype=int) 
            filament1=new[j+1:j+1+fil_head[2]]
            temp=filament1[0][0].split()

            x10=self.bound_cond(float(temp[-3]), l)
            y10=self.bound_cond(float(temp[-2]), l)
            z10=self.bound_cond(float(temp[-1]),l)
            r1=[]
            r2=[]
            r3=[]
            r4=[]
            r1.append([x10, y10, z10])

            for k in range(1, len(filament1), 1):
                temp=filament1[k][0].split(' ')
                x10=self.bound_cond(float(temp[-3]), l)
                y10=self.bound_cond(float(temp[-2]), l)
                z10=self.bound_cond(float(temp[-1]),l)
                r10=[x10, y10, z10]

                r1_old=np.array([r1[-1]])
                dis1=np.linalg.norm(r1_old-np.array(r10))

                if (len(r2)!=0):
                    r2_old=np.array([r2[-1]])
                    dis2=np.linalg.norm(r2_old-np.array(r10))    
                else:
                    dis2=l/3

                if (len(r3)!=0):
                    r3_old=np.array([r3[-1]])
                    dis3=np.linalg.norm(r3_old-np.array(r10))   
                else:
                    dis3=l/3

                if(dis1<l/2):
                    r1.append(r10)
                elif (dis2<l/2):
                    r2.append(r10)
                elif (dis3<l/2):
                    r3.append(r10)
                else:
                    r4.append(r10)

            r1=np.array(r1)
            r2=np.array(r2)
            r3=np.array(r3)
            r4=np.array(r4)

            R=[r1, r2, r3, r4]
            out.append(R)
            j=j+1+fil_head[2]
        return out
    
    def length(self,data, start, fil_num):
        ''' function returning an array of lengths of filaments'''
        l=self.l
        fil_length=np.zeros(fil_num, dtype=float)
        new=np.array(data[start:])
        j=0
        hello=[] #lenghts of filament segments

        for i in range(fil_num):
            fil_head=np.array(new[j][0].split(), dtype=int) 
            filament1=new[j+1:j+1+fil_head[2]]
            temp=filament1[0][0].split(' ')

            x10=self.bound_cond(float(temp[-3]), l)
            y10=self.bound_cond(float(temp[-2]), l)
            z10=self.bound_cond(float(temp[-1]),l)
            r1=[]
            r1.append([x10, y10, z10])
            d=0 #length of the filament

            for k in range(1, len(filament1), 1):
                temp=filament1[k][0].split()
                x10=self.bound_cond(float(temp[-3]), l)
                y10=self.bound_cond(float(temp[-2]), l)
                z10=self.bound_cond(float(temp[-1]),l)
                r10=[x10, y10, z10]

                r1_old=np.array([r1[-1]])
                dr=np.abs(r1_old-np.array(r10)) #positive definite distance between the points
                dr[dr>l/2]-=l #accounting for periodic boundary conditions
                d+=np.linalg.norm(dr) #increasing the length of the filament
                hello.append(np.linalg.norm(dr))
                r1.append(r10)

            j=j+1+fil_head[2]
            fil_length[i]=d
        return fil_length
    
    def get_crit(file, D):
        ''' gives the type and positions of critical points from the file for D dimensional space
        file: file of the type of NDskl_ascii'''

        data1=pd.read_csv(file,on_bad_lines='skip')
        start=0
        crit="[CRITICAL POINTS]"
        for i in range(len(data1)):
            line=np.array(data1[i:i+1])
            if crit in str(line):
                start=i+1
                break  

        ncrits=int(np.array(data1[start:start+1]))

        new=np.array(data1[start+1:])
        type_crit=np.zeros(ncrits, dtype=int)
        pos_crit=np.zeros([ncrits, D])
        ind_crit=0 
        start_crit=0
        while(ind_crit< ncrits):
            cri=np.array(new[start_crit][0].split(), dtype=float)
            type_crit[ind_crit]= int(cri[0])
            pos_crit[ind_crit]=cri[1:1+D]
            skip=int(new[start_crit+1][0])
            ind_crit+=1
            start_crit+=skip+2

        return type_crit, pos_crit
