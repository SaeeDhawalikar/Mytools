import numpy as np
import pandas as pd
import h5py
import re

# processes output of disperse.
# valid for non-periodic boxes
class Disperse:
    def __init__(self, l=300, D=3):
        self.l=l
        self.D=D

    def start_fil(self,file):
        ''' Gives the starting point to read filament data
        input:
        file: disperse skeleton output file
        output:
        Dictionary D1 with the keys
        start: starting point to read the filaments data
        nfil: number of filaments
        '''
        data=pd.read_csv(file,on_bad_lines='skip')
        start=0
        a="[FILAMENTS]"
        for i in range(len(data)):
            line=np.array(data[i:i+1])
            if a in str(line):
                start=i+2
        fil_num=int(np.array(data[start-1:start]))
        D1=dict()
        D1["start"]=start
        D1["nfil"]=fil_num
      
        return D1
    

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
    
    def get_crit(self,file):
        ''' gives the information about critical points in the file
        Inputs:
        file: file of the type of NDskl_ascii

        Outputs:
        Dictionary D1 with the following keys
        ncrit: number of critical points
        nmax: number of maxima
        nmin: number of mimina
        nsad: number of saddle points of order D-1
        nbif: number of bifurcation points

        type: type of the critical point (0 for minima, D for maxima) shape[N]
        pos_crit: positions of the critical points , shape [N, 3]
        nfil_crit: number of critical points associated to the given critical point
        valid: indices of valid points'''
        D=self.D

        data1=pd.read_csv(file,on_bad_lines='skip')
        start=0
        crit="[CRITICAL POINTS]"
        for i in range(len(data1)):
            line=np.array(data1[i:i+1])
            if crit in str(line):
                start=i+1
                break  

        ncrit=int(np.array(data1[start:start+1]))

        new=np.array(data1[start+1:])
        type_crit=np.zeros(ncrit, dtype=int)
        pos_crit=np.zeros([ncrit, D])
        nfil_crit=np.copy(type_crit)
        ind_crit=0 
        start_crit=0
        while(ind_crit< ncrit):
            cri=np.array(new[start_crit][0].split(), dtype=float)
            type_crit[ind_crit]= int(cri[0])
            pos_crit[ind_crit]=cri[1:1+D]
            nfil1=int(new[start_crit+1][0])
            nfil_crit[ind_crit]=nfil1
            ind_crit+=1
            start_crit+=nfil1+2


        # maxima
        ind=np.where(type_crit==D)[0]
        nmax=len(ind)
        # saddle
        ind=np.where(type_crit==D-1)[0]
        nsad=len(ind)
        #bifurcation
        ind=np.where(type_crit==D+1)[0]
        nbif=len(ind)
        # minima
        ind=np.where(type_crit==0)[0]
        nmin=len(ind)

        # valid critical points: the saddle and bifurcation points must be connected to atleast 2 filaments
        valid=[]
        for i in range(ncrit):
            if (type_crit[i]==D):
                valid.append(i)
            elif ((type_crit[i]==D+1)&(nfil_crit[i]>1)):
                valid.append(i)
            elif ((type_crit[i]==D-1)&(nfil_crit[i]>1)):
                valid.append(i)
        valid=np.array(valid)


        D1=dict()
        D1["ncrit"]=ncrit
        D1["pos_crit"]=pos_crit
        D1["type"]=type_crit
        D1["nmax"]=nmax
        D1["nmin"]=nmin
        D1["nsad"]=nsad
        D1["nbif"]=nbif
        D1["nfil_crit"]=nfil_crit
        D1["valid"]=valid
        return D1

    def plot_fil(self,file):
        """gives the data required for plotting the filaments
        inputs:
        file: NDskl_ascii file

        output:
        dictionary out with the following keys
        nfil: total number of filaments (valid+ invalid)
        nfil_valid: total number of valid filaments
        pos: list of arrays [P1, P2, ...], 
        where Pi are the positions of particles in the ith filament in the shape[Ni,D]"""
        out=dict()
        D=self.D
        D1=self.get_crit(file)

        valid=D1["valid"]
        start=0
        a="[FILAMENTS]"
        data=pd.read_csv(file,on_bad_lines='skip')
        for i in range(len(data)):
            line=np.array(data[i:i+1])
            if a in str(line):
                start=i+2
        fil_num=int(np.array(data[start-1:start]))
        out["nfil"]=fil_num

        # looping over the filaments
        pos=[]
        new=np.array(data[start:])
        j=0
        nfil_valid=0
        for i in range(fil_num):
            fil_head=np.array(new[j][0].split(), dtype=int) 
            print(fil_head)
            # indices of the critical points
            ind1=fil_head[0]
            ind2=fil_head[1]
            #keeping only valid filaments
            if ((ind1 in valid)&(ind2 in valid)):
                nfil_valid+=1
                filament1=new[j+1:j+1+fil_head[2]]
                po=np.zeros([fil_head[2], 3])
                for k in range(len(filament1)):
                    p=filament1[k][0].split()
                    po[k]=np.array(p, dtype=float)
                pos.append(po)   
            else:
                print("filament skipped, has critical end points", ind1, ind2)
            j=j+1+fil_head[2]
        out["pos"]=pos
        out["nfil_valid"]=nfil_valid
        return out
    
    def get_all(self, file):
        """ gives the all the data of interest related to the filaments and critical points.
        This data is not filtered.
        
        inputs: file: NDskl_ascii file
        """
        data=pd.read_csv(file,on_bad_lines='skip')
        D=3
        start=0
        a="[FILAMENTS]"
        for i in range(len(data)):
            line=np.array(data[i:i+1])
            if a in str(line):
                start=i+2
        fil_num=int(np.array(data[start-1:start]))
        D1=dict()
        D1["start"]=start # starting point of the filament
        D1["nfil"]=fil_num # number of filaments

        # critical point data
        start=0
        crit="[CRITICAL POINTS]"
        for i in range(len(data)):
            line=np.array(data[i:i+1])
            if crit in str(line):
                start=i+1
                break  

        ncrit=int(np.array(data[start:start+1]))

        new=np.array(data[start+1:])
        type_crit=np.zeros(ncrit, dtype=int)
        pos_crit=np.zeros([ncrit, D])
        nfil_crit=np.copy(type_crit)
        ind_crit=0 
        start_crit=0
        while(ind_crit< ncrit):
            cri=np.array(new[start_crit][0].split(), dtype=float)
            type_crit[ind_crit]= int(cri[0])
            pos_crit[ind_crit]=cri[1:1+D]
            nfil1=int(new[start_crit+1][0])
            nfil_crit[ind_crit]=nfil1 # number of filaments attached to that critical point
            ind_crit+=1
            start_crit+=nfil1+2


        # maxima
        ind=np.where(type_crit==D)[0]
        nmax=len(ind)
        # saddle
        ind=np.where(type_crit==D-1)[0]
        nsad=len(ind)
        #bifurcation
        ind=np.where(type_crit==D+1)[0]
        nbif=len(ind)
        # minima
        ind=np.where(type_crit==0)[0]
        nmin=len(ind)


        D1["ncrit"]=ncrit
        D1["pos_crit"]=pos_crit
        D1["type"]=type_crit
        D1["nmax"]=nmax
        D1["nmin"]=nmin
        D1["nsad"]=nsad
        D1["nbif"]=nbif
        D1["nfil_crit"]=nfil_crit #number of filaments attached to this critical point
        ###############################################################

        # filament data
        start=0
        a="[FILAMENTS]"
        for i in range(len(data)):
            line=np.array(data[i:i+1])
            if a in str(line):
                start=i+2
        fil_num=int(np.array(data[start-1:start])) # total number of filaments

        # looping over the filaments
        pos=[] # positions of all the points in the filamnets
        new=np.array(data[start:])
        j=0 # starting point in the file for a given filament

        ID_fil=np.arange(0, fil_num, 1) # filament id
        endfil1=np.zeros_like(ID_fil) #indices of critical points at the ends of the filament
        endfil2=np.zeros_like(ID_fil)
        for i in range(fil_num):
            fil_head=np.array(new[j][0].split(), dtype=int) 
            # indices of the critical points at the ends of the filaments
            endfil1[i]=fil_head[0]
            endfil2[i]=fil_head[1]

            filament1=new[j+1:j+1+fil_head[2]]
            po=np.zeros([fil_head[2], 3])
            for k in range(len(filament1)):
                p=filament1[k][0].split()
                po[k]=np.array(p, dtype=float)
            pos.append(po)   

            j=j+1+fil_head[2]


        D1["pos_fil"]=pos
        D1["ID_fil"]=ID_fil
        D1["endfil1"]=endfil1
        D1["endfil2"]=endfil2
        
        return D1
    
    def clean(self, D1, delloop=True):
        """ cleans the filaments, so that only those filaments which end on a maxima are retained.
        
        inputs:
        D1: dictionary that is the output of get_all function
        delloop: True if one wants to delete loops between saddle points and/or bifurcation points
                True by default
        
        outputs: dictionary D2 with the following keys:
        valid_crit: ids of valid critical points
        valid_fil: ids of valid filaments
        valid_end1, valid_end2: critical points at the ends of the valid filaments
        """
        dim=self.D
        ncrit=D1["ncrit"]
        ID_crit=np.arange(0, ncrit, 1) # ids of critical points
        new_nfil_crit=D1["nfil_crit"].copy() # number of filaments connected to the critical points, updated
        type_crit=D1["type"] # type of the critical point

        valid_fil=D1["ID_fil"].copy() # ids of the valid filaments
        valid_end1=D1["endfil1"].copy() # end point ids for the valid filaments
        valid_end2=D1["endfil2"].copy()
        
        # deleting the loops
        DEL=[]
        nloop=0
        if (delloop is True):
            for i in range(len(valid_fil)):
                e1=valid_end1[i]
                e2=valid_end2[i]

                if ((type_crit[e1]!=dim)&(type_crit[e2]!=dim)):
                    for j in range(i+1, len(valid_fil), 1):
                        E1=valid_end1[j]
                        E2=valid_end2[j]


                        if (((e1==E1)&(e2==E2))+((e1==E2)&(e2==E1))):
                            nloop+=1
                            DEL.extend([i, j])
                            new_nfil_crit[e1]-=2 # reducing the number of filaments connected to that particular point
                            new_nfil_crit[e2]-=2

        DEL=np.array(DEL)
        print("\n\n critical points to be deleted del:", DEL)
        if (len(DEL)>0):
            valid_fil=np.delete(valid_fil, DEL)
            valid_end1=np.delete(valid_end1, DEL)
            valid_end2=np.delete(valid_end2, DEL)

        valid_crit=np.union1d(valid_end1, valid_end2)
        print("\nnumber of loops deleted:", nloop)
        print("number of disperse filaments deleted:",len(DEL) )
        # note that this does not account for more than 2 filaments connecting the same set of points
        
        ################################################################################################
        # deleting unconnected filaments
        ndel=1 # number of deleted points in the previous iteration
        while (ndel>0):
            dele=[] # indices of critical points to be deleted
            for i in range(len(valid_crit)):
                this_crit=valid_crit[i] # id of the current critical point
                # always retain maxima, delete saddle and bifurcation points if only 1 filament connected
                if ((type_crit[this_crit]!=dim)&(new_nfil_crit[this_crit]<2)):
                    dele.append(i)
                    new_nfil_crit[this_crit]-=1 # decreasing the number of filaments connected to this point by 1
                    # finding the filament linked to this point
                    if (this_crit in valid_end1):
                        ind=np.where(valid_end1==this_crit)[0]
                        other=valid_end2[ind] # index of the critical point at the other end of the filament
                        new_nfil_crit[other]-=1

                        valid_fil=np.delete(valid_fil, ind) #deleting the corresponding filament
                        valid_end1=np.delete(valid_end1, ind) 
                        valid_end2=np.delete(valid_end2, ind) 

                    elif (this_crit in valid_end2):
                        ind=np.where(valid_end2==this_crit)[0]
                        other=valid_end1[ind] # index of the critical point at the other end of the filament
                        new_nfil_crit[other]-=1

                        valid_fil=np.delete(valid_fil, ind) #deleting the corresponding filament
                        valid_end1=np.delete(valid_end1, ind) 
                        valid_end2=np.delete(valid_end2, ind)
                        

            print("\n indices of the critical points deleted",valid_crit[dele])
            # deleting the unwanted critical points      
            valid_crit=np.delete(valid_crit, dele)
            ndel=len(dele) # number of critical points deleted in this iteration


        D2=dict()
        D2["valid_crit"]=valid_crit
        D2["valid_fil"]= valid_fil
        D2["valid_end1"]=valid_end1
        D2["valid_end2"]=valid_end2
        
        return D2
    
    def join_fila(self, D1, D2 ):
        """ joins the filaments starting from one node, till it reaches the other node.
        Not valid in case the filaments bifurcate
        
        inputs:
        D1: output dictionary of get_all function
        D2: output dictionary of clean function
        
        output:
        POS_F: [N, 3] array giving the cartesian coordiantes of all the particles of the filament"""
        validend1=D2["valid_end1"]
        validend2=D2["valid_end2"]
        valid_fil=D2["valid_fil"]

        pos=D1["pos_fil"]
        type_crit=D1["type"]

        # choosing the starting end point
        validendtype=type_crit[validend2]

        I=0 # which filament to start with
        rem_fil=valid_fil.copy().tolist() # list of remaining filaments
        dim=self.D

        final_fil=[] # positions of all the points in the joined filament
        for i in range(len(validendtype)):
            if (validendtype[i]==dim):
                I=rem_fil[i] # index of the filament
                break

        P1=np.flip(pos[I], axis=0) # flipping so the the node is at the beginning
        final_fil.append(P1) # flipped so that the starting is the node
        rem_fil.remove(I)

        print("first filament taken:",I)
        print("remaining filaments:", rem_fil)
        END=validend1[i]
        validend1=np.delete(validend1, i)
        validend2=np.delete(validend2, i)

        print("\n end points of the first filament:",P1[0], P1[-1])

        while(len(rem_fil)>0):
            if (END in validend1):
                ind=int(np.where(validend1==END)[0])
                END=validend2[ind]
                validend1=np.delete(validend1, ind)
                validend2=np.delete(validend2, ind)
                temp=rem_fil[ind]
                p_add=pos[temp]
                final_fil.append(p_add)
                rem_fil.remove(temp)
                print("end points of the new added filament:",pos[temp][0], pos[temp][-1])


            elif (END in validend2):
                ind=int(np.where(validend2==END)[0])
                END=validend1[ind]
                validend1=np.delete(validend1, ind)
                validend2=np.delete(validend2, ind)
                temp=rem_fil[ind]
                rem_fil.remove(temp)
                p_add=np.flip(pos[temp], axis=0)
                final_fil.append(p_add)
                print("end points of the new added filament, flipped:",p_add[0], p_add[-1])
        final_fil=np.concatenate(final_fil, axis=0)
        return final_fil

