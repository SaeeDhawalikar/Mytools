import numpy as np
import h5py
import pandas as pd
import sys

class gadget():
    def __init__(self, file_array, l=300):
        self.file_array=file_array
        self.l=l

    def W(self,a,b,c, nbox):
        ''' gives the weight of the box given its x y and z indices and the number of cells along an axis''' 
        a=a%nbox #periodic boundary conditions
        b=b%nbox
        c=c%nbox
        return (a*nbox**2+b*nbox+c)

##############################################################################################


    def get_pos(self):
        '''This function gives the positions of DM particles in the given gadget files
        file_array must be a list of the files containing the gadget output'''

        #loading the data
        prtcl_types = ["Gas","Halo","Disk",  "Bulge", "Stars", "Bndry"]
        prtcl_type="Halo"
        file = open(self.file_array[0],'rb')

        header_size = np.fromfile(file, dtype=np.uint32, count=1)
        N_prtcl_thisfile = np.fromfile(file, dtype=np.uint32, count=6)    ## The number of particles of each type present in the file

        file.seek(256+8, 0)
        position_block_size = np.fromfile(file, dtype = np.int32, count =1)[0]

        N_prtcl = N_prtcl_thisfile[1] #halo particles
        i = 0
        while prtcl_types[i] != prtcl_type:
            file.seek(N_prtcl_thisfile[i]*3*4, 1)
            i += 1
        posd = np.fromfile(file, dtype = np.float32, count = N_prtcl*3)
        posd = posd.reshape((N_prtcl, 3))  
        Pos=pd.DataFrame(posd)
        file.close()

        #extending the data
        for j in range(1,len(self.file_array),1):
                file = open(self.file_array[j],'rb')

                header_size = np.fromfile(file, dtype=np.uint32, count=1)
                N_prtcl_thisfile = np.fromfile(file, dtype=np.uint32, count=6) # The number of particles of each type present in the file

                file.seek(256+8, 0)
                position_block_size = np.fromfile(file, dtype = np.int32, count =1)[0]


                N_prtcl = N_prtcl_thisfile[1]
                i = 0
                while prtcl_types[i] != prtcl_type:
                    file.seek(N_prtcl_thisfile[i]*3*4, 1)
                    i += 1

                posd = np.fromfile(file, dtype = np.float32, count = N_prtcl*3)
                posd = posd.reshape((N_prtcl, 3))
                df_x=pd.DataFrame(posd)
                Pos=pd.DataFrame(np.concatenate([Pos, df_x]))
                file.close()

        Pos=np.array(Pos)
        return Pos

    def get_vel(self):
        '''This function gives the velocities of DM particles in the given gadget files
        g must be a list of the files containing the gadget output'''

        #loading the velocity data
        prtcl_types = ["Gas","Halo","Disk",  "Bulge", "Stars", "Bndry"]
        prtcl_type="Halo"
        Vel1=[]
        for k in range(len(self.file_array)):
            file = open(self.file_array[k],'rb')

            header_size = np.fromfile(file, dtype=np.uint32, count=1)
            N_prtcl_thisfile = np.fromfile(file, dtype=np.uint32, count=6)    ## The number of particles of each type present in the file


            file.seek(256+8+8 + int(N_prtcl_thisfile.sum())*3*4, 0)
            velocity_block_size = np.fromfile(file, dtype = np.int32, count =1)[0]
            i = 0
            while prtcl_types[i] != prtcl_type:
                file.seek(N_prtcl_thisfile[i]*3*4, 1)
                i += 1
            N_prtcl = N_prtcl_thisfile[i]
            veld = np.fromfile(file, dtype = np.float32, count = N_prtcl*3) 
            veld = veld.reshape((N_prtcl, 3))

            Vel1.append(veld)
            file.close()
        Vel=np.concatenate([Vel1[0], Vel1[1], Vel1[2], Vel1[3], Vel1[4], Vel1[5], Vel1[6], Vel1[7]])
        return Vel


    def split(self,Pos=None, nbox=30):
        """ This function gives the indices of the particles such that they are arranged into small sub-boxes"""
        lbox=self.l/nbox
        edge=np.array(np.linspace(lbox, self.l, nbox)) #left edge of the sub-boxes
        if Pos is None:
            Pos=self.get_pos()
        rind=np.array(Pos/lbox, dtype=int) #(i,j,k) index of the box that the particle belongs to
        weight=rind[:,0]*nbox**2+rind[:,1]*nbox+rind[:,2] # index of the box, when the 3D array is flattened
        rind=0  #to save space

        sort=weight.argsort()  #finally, one has to do Pos[sort], Vel[sort]
        weight=weight[sort]
        Num_W=np.bincount(weight)  #number of particles in each sub-box
        weight=0
        Num_cum=np.cumsum(Num_W) #number of particles below the given sub-box 
        #make sure that there are non-zero particles per sub-box, otherwise Num-cum has dimensions different from number of sub-boxes, and the code will fial.
        Num_cum=np.roll(Num_cum, 1)
        Num_cum[0]=0
        
        return sort, Num_cum, Num_W


#####################################################################################################################
class halos():
    def __init__(self, file):
        self.file=file
        
        with open(self.file) as f:
            head=f.readline()
        header=head[1:].split()
        self.data_halo=pd.read_csv(self.file, sep="\s+", comment="#",names=header)
        
    def keys(self):
        return self.data_halo.keys()
    
    def data(self):
        return self.data_halo
    
#     def cond(self, subhalos=False, QE=0.5, M0=None, N=None, M1=None, M2=None):
#         """ returns the indices of the clean data with:
#         subhalo=False removes the subhalos
#         QE gives the condition of virialisation
#         M1 gives the median mass of halos
#         N gives the number of halos on either side
#         or M1, M2 give the mass range"""
        
#         ind1=0
#         indi=0
#         ind_bounds=0
#         if (self.l==300):
#             TbyU_max = 0.5*(1+QE)
#             TbyU_min = np.max([0.0,0.5*(1-QE)])
#             cond_clean = ((self.data["T/|U|"] < TbyU_max) & (TbyU_min < self.data["T/|U|"])) #virial condition
            
#             if (subhalos==False):
#                 cond_clean = cond_clean & (self.data["PID"] == -1) #removing sub-halo
                
            
#             data1=self.data[cond_clean]
#             Mvir=np.array(data1["Mvir"][:])
            
#             if (M0 is not None):
#                 ind1=np.where((Mvir>M0/2)&(Mvir<2*M0))[0]
#                 Mvir1=Mvir[ind1]

#                 arg=np.argsort(Mvir1)
#                 Mvir1=Mvir1[arg]
#                 temp=np.abs(Mvir1-M0) #temporary array
#                 med1=np.where(temp==np.min(temp))[0]
#                 med=int(np.median(med1)) #index of the median halo
#                 indi=arg[med-N:med+N+1]
                
#             elif(M1 is not None):
#                 ind_bounds=np.where((Mvir>M1)&(Mvir<M2))[0]
                
                
#         elif (self.l==150):
#             TbyU_max = 0.5*(1+QE)
#             TbyU_min = np.max([0.0,0.5*(1-QE)])
#             cond_clean = ((self.data["T/|U|"] < TbyU_max) & (TbyU_min < self.data["T/|U|"])) #virial condition

#             if (subhalos==False):
#                 cond_clean = cond_clean & (self.data["PID"] == -1) #removing sub-halo


#             data1=self.data[cond_clean]
#             Mvir=np.array(data1["Mvir"][:])

#             if (M0 is not None):
#                 ind1=np.where((Mvir>M0/2)&(Mvir<2*M0))[0]
#                 Mvir1=Mvir[ind1]

#                 arg=np.argsort(Mvir1)
#                 Mvir1=Mvir1[arg]
#                 temp=np.abs(Mvir1-M0) #temporary array
#                 med1=np.where(temp==np.min(temp))[0]
#                 med=int(np.median(med1)) #index of the median halo
#                 indi=arg[med-N:med+N+1]

#             elif(M1 is not None):
#                 ind_bounds=np.where((Mvir>M1)&(Mvir<M2))[0]



#         return cond_clean, ind1, indi, ind_bounds
    
    
                
            

        
    
    

# 
# Rvir=np.array(data["Rvir"][:])/1e3 #units Mpc/h
# x_halo=np.array(data["X"] )#units Mpc/h
# y_halo=np.array(data["Y"])
# z_halo=np.array(data["Z"])

# vx_halo=np.array(data["VX"]) #units km/s
# vy_halo=np.array(data["VY"])
# vz_halo=np.array(data["VZ"])
# #sigv=np.array(data["Vrms"]) #km/s

# Pos_halo=np.array([x_halo, y_halo, z_halo])
# Vel_halo=np.array([vx_halo, vy_halo, vz_halo])

# #tidal anisotropy data
# file="/scratch/aseem/halos/su1024/delta0.0/r%d/out_1.vahc"%argument
# colnames=['haloID', 'lam1_R2R200b', 'lam2_R2R200b', 'lam3_R2R200b',
#        'lam1_R4R200b', 'lam2_R4R200b', 'lam3_R4R200b', 'lam1_R6R200b',
#        'lam2_R6R200b', 'lam3_R6R200b', 'lam1_R8R200b', 'lam2_R8R200b',
#        'lam3_R8R200b', 'lam1_R2Mpch', 'lam2_R2Mpch', 'lam3_R2Mpch',
#        'lam1_R3Mpch', 'lam2_R3Mpch', 'lam3_R3Mpch', 'lam1_R5Mpch',
#        'lam2_R5Mpch', 'lam3_R5Mpch', 'lamH1_R3Mpch', 'lamH2_R3Mpch',
#        'lamH3_R3Mpch', 'lamH1_R5Mpch', 'lamH2_R5Mpch', 'lamH3_R5Mpch', 'b1',
#        'b1wtd', "#"]
# data=pd.read_csv(file, sep=" ", header=1, names=colnames)
# lam1=np.array(data["lam1_R4R200b"])
# lam2=np.array(data["lam2_R4R200b"])
# lam3=np.array(data["lam3_R4R200b"])
# delta_R=lam1+lam2+lam3
# qR=np.abs((((lam3-lam2)**2+(lam3-lam1)**2+(lam2-lam1)**2)/2)**0.5)
# alpha=qR/(1+delta_R)


# ##########################################################################################################
# #selecting 2N+1 halos with median M1
# #Setting scales of interest by hand to match approximate values of median mass halos

# #keeping halos with masses lying between M1/2 to 2*M1 a
# ind1=np.where((Mvir>M1/2)&(Mvir<2*M1))[0]
# Mvir1=Mvir[ind1]
# Rvir1=Rvir[ind1]
# Pos_halo1=Pos_halo[:, ind1]
# Vel_halo1=Vel_halo[:, ind1]
# alpha1=alpha[ind1]
# #arranging in ascending order in halo mass
# arg=np.argsort(Mvir1)

# Mvir1=Mvir1[arg]
# Rvir1=Rvir1[arg]
# Pos_halo1=Pos_halo1[:,arg]
# Vel_halo1=Vel_halo1[:, arg]
# alpha1=alpha1[arg]
# # choosing the median halo and keeping N halos on either side
# temp=np.abs(Mvir1-M1) #temporary array
# med1=np.where(temp==np.min(temp))[0]
# med=int(np.median(med1)) #index of the median halo


# Mvir1=Mvir1[med-N:med+N+1]
# Rvir1=Rvir1[med-N:med+N+1]
# Pos_halo1=Pos_halo1[:, med-N:med+N+1]
# Vel_halo1=Vel_halo1[:, med-N:med+N+1]
# alpha1=alpha1[med-N:med+N+1]

# sigv1=6.5567*np.sqrt(Mvir1/Rvir1)/np.sqrt(3)

# #applying the alpha constriant
# indh=np.where(alpha1>=alpha_h)[0]
# indl=np.where(alpha1<=alpha_l)[0]

# Mvir_h=Mvir1[indh]
# Rvir_h=Rvir1[indh]
# Pos_halo_h=Pos_halo1[:, indh]
# Vel_halo_h=Vel_halo1[:, indh]
# alpha_h=alpha1[indh]
# sigv_h=sigv1[indh]

# Mvir_l=Mvir1[indl]
# Rvir_l=Rvir1[indl]
# Pos_halo_l=Pos_halo1[:, indl]
# Vel_halo_l=Vel_halo1[:, indl]
# alpha_l=alpha1[indl]
# sigv_l=sigv1[indl]

# #removing unwanted arrays 
# Rvir1=0
# Mvir1=0
# Vel_halo1=0
# Pos_halo1=0
# alpha1=0
# sigv1=0

# ######################################################################################################
# ##2d pdfs
# #tubes have radius 4*Rvir0

# dx_dimless=4 # half width of the tube in terms of Rvir
# dy_dimless=4
# dz_dimless=30
# vmax_dimless=6 #maximum velocity plotted in units of 1d sigma_vir

# vbins=100
# zbins=100

# #bin_n=9 #number of cuboids the tube is split into. here, length is slightly larger than width
# #cube_bins=np.linspace(-dz_dimless, dz_dimless, bin_n+1) #edges of the sub-cubes in Mpc/h
# #vz_hist=np.zeros([bin_n, vbins], dtype=float) #array to store the velocity histograms

# #creating arrays to store v_z histograms
# v_z_hist_h=np.zeros([3,vbins, zbins], dtype=float) #2d pdfs
# x_z_hist_h=np.zeros([zbins,zbins], dtype=float) #for stacked density slice
# z_bins=np.array(np.linspace(-dz_dimless, dz_dimless, zbins+1)) #bins along the tube for 2d histogram and density projections
# x_bins=np.array(np.linspace(-dx_dimless, dx_dimless, zbins+1))  #position bins for projected density
# v_bins=np.array(np.linspace(-vmax_dimless, vmax_dimless, vbins+1)) #velocity bins for 2d histograms

# for i in range(len(Mvir_h)):
#     r0=Pos_halo_h[:, i]
#     v0=Vel_halo_h[:,i]
#     Rvir0=Rvir_h[i] #halo Rvir in Mpc/h
#     sigv0=sigv_h[i]
    
#     vmax=vmax_dimless*sigv0
#     dx=dx_dimless*Rvir0
#     dy=dy_dimless*Rvir0
#     dz=dz_dimless*Rvir0
#     dr=np.array([dx, dy, dz], dtype=float)
#     ###########################################################################
#     #sub boxes to be considered
#     ind0=np.array(r0/lbox, dtype=int)
#     indmax=np.array(((r0+dr)%l)/lbox, dtype=int)
#     indmin=np.array(((r0-dr)%l)/lbox, dtype=int)

#     indmin[np.abs(indmin-ind0)>nbox/2]=indmin[np.abs(indmin-ind0)>nbox/2]-nbox  #shifting according to pbc
#     indmax[np.abs(indmax-ind0)>nbox/2]=indmax[np.abs(indmax-ind0)>nbox/2]+nbox

#     xind=np.arange(indmin[0], indmax[0]+1,1)
#     yind=np.arange(indmin[1], indmax[1]+1,1)
#     zind=np.arange(indmin[2], indmax[2]+1,1)

#     X, Y, Z= np.meshgrid(xind, yind, zind)
#     X=X.flatten()
#     Y=Y.flatten()
#     Z=Z.flatten()
#     weights=W(X, Y, Z, nbox)
#     #####################################################################################
    
#     weight0=weights[0]
#     Pos1=Pos[Num_cum[weight0]:Num_cum[weight0]+Num_W[weight0],:]
#     Vel1=Vel[Num_cum[weight0]:Num_cum[weight0]+Num_W[weight0],:]

#     ind=np.where((np.abs(Pos1[:,0]-r0[0])<dx)+(l-np.abs(Pos1[:,0]-r0[0])<dx))[0]
#     Pos1=Pos1[ind,:]
#     Vel1=Vel1[ind,:]

#     ind=np.where((np.abs(Pos1[:,1]-r0[1])<dy)+(l-np.abs(Pos1[:,1]-r0[1])<dy))[0]
#     Pos2=Pos1[ind, :]
#     Vel2=Vel1[ind, :]

#     ind=np.where((np.abs(Pos2[:,2]-r0[2])<dz)+(l-np.abs(Pos2[:,2]-r0[2])<dz))[0]
#     Pos2=Pos2[ind, :]
#     Vel2=Vel2[ind, :]

#     for b in range(1,len(weights),1):
#         Pos1=Pos[Num_cum[weights[b]]:Num_cum[weights[b]]+Num_W[weights[b]],:]
#         Vel1=Vel[Num_cum[weights[b]]:Num_cum[weights[b]]+Num_W[weights[b]],:]

#         ind=np.where((np.abs(Pos1[:,0]-r0[0])<dx)+(l-np.abs(Pos1[:,0]-r0[0])<dx))[0]
#         Pos1=Pos1[ind,:]
#         Vel1=Vel1[ind,:]

#         ind=np.where((np.abs(Pos1[:,1]-r0[1])<dy)+(l-np.abs(Pos1[:,1]-r0[1])<dy))[0]
#         Pos1=Pos1[ind, :]
#         Vel1=Vel1[ind, :]

#         ind=np.where((np.abs(Pos1[:,2]-r0[2])<dz)+(l-np.abs(Pos1[:,2]-r0[2])<dz))[0]
#         Pos1=Pos1[ind, :]
#         Vel1=Vel1[ind, :]

#         Pos2=np.concatenate([Pos2,Pos1],axis=0)
#         Vel2=np.concatenate([Vel2,Vel1],axis=0)

#     Pos2=(np.subtract(np.subtract(Pos2, (r0-l/2))%l, l/2))/Rvir0
#     Vel2=(np.subtract(Vel2, v0))/sigv0
        
#     Pos1=Pos2
#     Vel1=Vel2
#     Vel2=0
#     Pos2=0
#    #changing the limits in velocity pdf so as to take into account all the particles
#     v_bins[0]=np.min([np.min(Vel1), -vmax_dimless])
#     v_bins[len(v_bins)-1]=np.max([np.max(Vel1), vmax_dimless])
 
#     #2d pdfs
#     for j in range(3):
#         H, v_edge, z_edge=np.histogram2d(Vel1[:,j],Pos1[:,2], bins=[v_bins, z_bins]) 
#         #not normalised, so that mean is weighted
#         v_z_hist_h[j]+=H
    
#     #density slices    
#     J, x_edge, z_edge=np.histogram2d(Pos1[:,0], Pos1[:,2], bins=[x_bins, z_bins])
#     x_z_hist_h+=J
    
# #    #velocity pdfs
# #    for k in range(len(cube_bins)-1):
# #        ind=np.where((Pos1[:,2]>=cube_bins[k])&(Pos1[:,2]<cube_bins[k+1]))
# #        vz_hist[k]+=np.histogram(Vel1[ind,2], bins=v_bins)[0]
        
        
# DF=pd.read_csv("/mnt/home/student/csaee/perl5/v_z_h.csv", header=None)
# temp=DF.values
# temp1=temp.reshape(3,vbins,zbins)+v_z_hist_h
# temp2=temp1.reshape(3*vbins, zbins)
# ar=pd.DataFrame(temp2)
# ar.to_csv("/mnt/home/student/csaee/perl5/v_z_h.csv", index=False, header=False)

# DF1=pd.read_csv("/mnt/home/student/csaee/perl5/x_z_h.csv", header=None)
# ar1=pd.DataFrame(DF1.values+x_z_hist_h)
# ar1.to_csv("/mnt/home/student/csaee/perl5/x_z_h.csv", index=False, header=False)

# #DF2=pd.read_csv("/mnt/home/student/csaee/perl5/vz.csv", header=None)
# #ar2=pd.DataFrame(DF2.values+vz_hist)
# #ar2.to_csv("/mnt/home/student/csaee/perl5/vz.csv", index=False, header=False)

# Mlim=pd.read_csv("Mlim_h.csv", header=None).values
# Mlim[argument-1,0]=np.min(Mvir_h)
# Mlim[argument-1,1]=np.max(Mvir_h)
# Mlim[argument-1, 2]=i
# hi=pd.DataFrame(Mlim)
# hi.to_csv("Mlim_h.csv", index=False, header=False)
# ########################################################################################################################################
# v_z_hist_l=np.zeros([3,vbins, zbins], dtype=float) #2d pdfs
# x_z_hist_l=np.zeros([zbins,zbins], dtype=float) #for stacked density slice

# for i in range(len(Mvir_l)):
#     r0=Pos_halo_l[:, i]
#     v0=Vel_halo_l[:,i]
#     Rvir0=Rvir_l[i] #halo Rvir in Mpc/h
#     sigv0=sigv_l[i]
    
#     vmax=vmax_dimless*sigv0
#     dx=dx_dimless*Rvir0
#     dy=dy_dimless*Rvir0
#     dz=dz_dimless*Rvir0
#     dr=np.array([dx, dy, dz], dtype=float)
#     ###########################################################################
#     #sub boxes to be considered
#     ind0=np.array(r0/lbox, dtype=int)
#     indmax=np.array(((r0+dr)%l)/lbox, dtype=int)
#     indmin=np.array(((r0-dr)%l)/lbox, dtype=int)

#     indmin[np.abs(indmin-ind0)>nbox/2]=indmin[np.abs(indmin-ind0)>nbox/2]-nbox  #shifting according to pbc
#     indmax[np.abs(indmax-ind0)>nbox/2]=indmax[np.abs(indmax-ind0)>nbox/2]+nbox

#     xind=np.arange(indmin[0], indmax[0]+1,1)
#     yind=np.arange(indmin[1], indmax[1]+1,1)
#     zind=np.arange(indmin[2], indmax[2]+1,1)

#     X, Y, Z= np.meshgrid(xind, yind, zind)
#     X=X.flatten()
#     Y=Y.flatten()
#     Z=Z.flatten()
#     weights=W(X, Y, Z, nbox)
#     #####################################################################################
    
#     weight0=weights[0]
#     Pos1=Pos[Num_cum[weight0]:Num_cum[weight0]+Num_W[weight0],:]
#     Vel1=Vel[Num_cum[weight0]:Num_cum[weight0]+Num_W[weight0],:]

#     ind=np.where((np.abs(Pos1[:,0]-r0[0])<dx)+(l-np.abs(Pos1[:,0]-r0[0])<dx))[0]
#     Pos1=Pos1[ind,:]
#     Vel1=Vel1[ind,:]

#     ind=np.where((np.abs(Pos1[:,1]-r0[1])<dy)+(l-np.abs(Pos1[:,1]-r0[1])<dy))[0]
#     Pos2=Pos1[ind, :]
#     Vel2=Vel1[ind, :]

#     ind=np.where((np.abs(Pos2[:,2]-r0[2])<dz)+(l-np.abs(Pos2[:,2]-r0[2])<dz))[0]
#     Pos2=Pos2[ind, :]
#     Vel2=Vel2[ind, :]

#     for b in range(1,len(weights),1):
#         Pos1=Pos[Num_cum[weights[b]]:Num_cum[weights[b]]+Num_W[weights[b]],:]
#         Vel1=Vel[Num_cum[weights[b]]:Num_cum[weights[b]]+Num_W[weights[b]],:]

#         ind=np.where((np.abs(Pos1[:,0]-r0[0])<dx)+(l-np.abs(Pos1[:,0]-r0[0])<dx))[0]
#         Pos1=Pos1[ind,:]
#         Vel1=Vel1[ind,:]

#         ind=np.where((np.abs(Pos1[:,1]-r0[1])<dy)+(l-np.abs(Pos1[:,1]-r0[1])<dy))[0]
#         Pos1=Pos1[ind, :]
#         Vel1=Vel1[ind, :]

#         ind=np.where((np.abs(Pos1[:,2]-r0[2])<dz)+(l-np.abs(Pos1[:,2]-r0[2])<dz))[0]
#         Pos1=Pos1[ind, :]
#         Vel1=Vel1[ind, :]

#         Pos2=np.concatenate([Pos2,Pos1],axis=0)
#         Vel2=np.concatenate([Vel2,Vel1],axis=0)

#     Pos2=(np.subtract(np.subtract(Pos2, (r0-l/2))%l, l/2))/Rvir0
#     Vel2=(np.subtract(Vel2, v0))/sigv0
        
#     Pos1=Pos2
#     Vel1=Vel2
#     Vel2=0
#     Pos2=0
#    #changing the limits in velocity pdf so as to take into account all the particles
#     v_bins[0]=np.min([np.min(Vel1), -vmax_dimless])
#     v_bins[len(v_bins)-1]=np.max([np.max(Vel1), vmax_dimless])
 
#     #2d pdfs
#     for j in range(3):
#         H, v_edge, z_edge=np.histogram2d(Vel1[:,j],Pos1[:,2], bins=[v_bins, z_bins]) 
#         #not normalised, so that mean is weighted
#         v_z_hist_l[j]+=H
    
#     #density slices    
#     J, x_edge, z_edge=np.histogram2d(Pos1[:,0], Pos1[:,2], bins=[x_bins, z_bins])
#     x_z_hist_l+=J
    
# #    #velocity pdfs
# #    for k in range(len(cube_bins)-1):
# #        ind=np.where((Pos1[:,2]>=cube_bins[k])&(Pos1[:,2]<cube_bins[k+1]))
# #        vz_hist[k]+=np.histogram(Vel1[ind,2], bins=v_bins)[0]
        
        
# DF=pd.read_csv("/mnt/home/student/csaee/perl5/v_z_l.csv", header=None)
# temp=DF.values
# temp1=temp.reshape(3,vbins,zbins)+v_z_hist_l
# temp2=temp1.reshape(3*vbins, zbins)
# ar=pd.DataFrame(temp2)
# ar.to_csv("/mnt/home/student/csaee/perl5/v_z_l.csv", index=False, header=False)

# DF1=pd.read_csv("/mnt/home/student/csaee/perl5/x_z_l.csv", header=None)
# ar1=pd.DataFrame(DF1.values+x_z_hist_l)
# ar1.to_csv("/mnt/home/student/csaee/perl5/x_z_l.csv", index=False, header=False)

# #DF2=pd.read_csv("/mnt/home/student/csaee/perl5/vz.csv", header=None)
# #ar2=pd.DataFrame(DF2.values+vz_hist)
# #ar2.to_csv("/mnt/home/student/csaee/perl5/vz.csv", index=False, header=False)

# Mlim=pd.read_csv("Mlim_l.csv", header=None).values
# Mlim[argument-1,0]=np.min(Mvir_l)
# Mlim[argument-1,1]=np.max(Mvir_l)
# Mlim[argument-1, 2]=i
# hi=pd.DataFrame(Mlim)
# hi.to_csv("Mlim_l.csv", index=False, header=False)



