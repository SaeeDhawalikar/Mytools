import numpy as np

class estimate_profile(object):
    """ estimates the 2D, logitudinal and radial profiles of filaments given the phase space information of the 
    particles of the box, a discretely sampled spine. """
    def __init__(self, lbox=None, pbc=False, mp=1):
        """ Inputs:
        lbox: box size (if not given, pbc cannot be applied)
        pbc: periodic boundary conditions. Default value is False
        mp: particle mass. Default value is 1 (i.e., density will be number density)"""
        self.l=lbox
        self.mp=mp
        self.pbc=pbc
        self.B=np.array([0,0,1], dtype=float) #Unit vector along the z-direction (filament axis)
        
    def rotate(self,pos, vec1, vec2):
        '''rotates the given position vectors pos so that vec1 has coordinates vec2 in the new coordinate system
        pos: vectors to be rotated with shape[3,N]
        vec1: coordinates of the vector in original coordinate system
        vec2: coordinates of the vector in new coordinate system'''

        dc=vec1/np.linalg.norm(vec1)
        B=vec2/np.linalg.norm(vec2)

        V=np.cross(dc, B)
        S=np.linalg.norm(V)
        C=np.dot(dc, B)

        Vx=np.array([[0, -V[2], V[1]],[V[2], 0, -V[0]], [-V[1], V[0],0]], dtype=float)
        I=np.array([[1,0,0], [0,1,0], [0,0,1]], dtype=float)

        if any(V):
            R=I+Vx+np.matmul(Vx, Vx)/(1+C) #rotation matrix  
        else:
            R=I #when the two axes align

        Pos_rot=np.matmul(R, pos)
        return Pos_rot
        
    def create_rbins(self,rmin, rmax, dlogr=None, dr=None, N_r=None, log=True):
        """ creates linearly or logarithmically spaced radial bins.
        
        Inputs:
        rmin, rmax: minimum and maximum values of radial bins
        dlogr: logarithmic spacing 
        dr: linear spacing
        Nr: number of bins [specify either dlogr (dr) or N_r in case of log (linear) spacing.]
        If both are specified, dlogr will be chosen, and N_r will be ignored.
        If none are specified, N_r will be set to 10
        
        Outputs:
        Dictionary D with keys:
        "rbins": end points of the bins
        "rmid": mid point of the bins
        """
        
        if (log is True):
            if (dr is not None):
                print("logarithmic binning is chosen, but linear spacing is specified!!")
                if(N_r is not None):
                    print("choosing the given value of Nr instead")
                else:
                    N_r=10
                    print("choosing Nr=10 instead")
                    
            if ((dlogr is not None)&(N_r is not None)):
                print("both dlogr and Nr are specified!! \n choosing dlogr")
            elif ((dlogr is None)&(N_r is None)):
                print("Neither dlogr nor Nr are specifield!! \n setting Nr to 10")
                N_r=10
                
                
            # creating the bins    
            if (dlogr is not None):
                rbins=10**(np.arange(np.log10(rmin), np.log10(rmax)+dlogr, dlogr))
            elif (N_r is not None):
                rbins=np.logspace(np.log10(rmin), np.log10(rmax), N_r+1)
                
            # mid points of the bins
            rmid=np.zeros(len(rbins)-1)
            for i in range(len(rmid)):
                rmid[i]=np.sqrt(rbins[i]*rbins[i+1])
                
    
        elif (log is False):
            if (dlogr is not None):
                print("linear binning is chosen, but logarithmic spacing is specified!!")
                if(N_r is not None):
                    print("choosing the given value of Nr instead")
                else:
                    N_r=10
                    print("choosing Nr=10 instead")
                    
            if ((dr is not None)&(N_r is not None)):
                print("both dr and Nr are specified!! \n choosing dr")
            elif ((dr is None)&(N_r is None)):
                print("Neither dr nor Nr are specifield!! \n setting Nr to 10")
                N_r=10
                
            # creating the bins    
            if (dr is not None):
                rbins=np.arange(rmin, rmax+dr, dr)
            elif (N_r is not None):
                rbins=np.linspace(rmin,rmax, N_r+1)
                
            # mid points of the bins
            rmid=np.zeros(len(rbins)-1)
            for i in range(len(rmid)):
                rmid[i]=(rbins[i]+rbins[i+1])/2
        # creating the output dictionary
        D=dict()
        D["rbins"]=rbins
        D["rmid"]=rmid
        return D
    
    def create_zbins(self,lfil, N_z=None):
        """ creates linearly spaced bins along the axis of the filament.
        Inputs:
        lfil: length of the filament
        N_z: number of bins along the axis of the filament. Default value is 20
        
        Outputs:
        Dictionary D with keys:
        "zbins": end points of the bins
        "zmid": mid point of the bins"""
        if (N_z is None):
            print("Nz not specified, setting Nz=20")
            N_z=20
        
        zbins=np.linspace(0, lfil, N_z+1)
        zmid=np.zeros(N_z)
        for i in range(N_z):
            zmid[i]=(zbins[i]+zbins[i+1])/2
        D=dict()
        D["zbins"]=zbins
        D["zmid"]=zmid
        return D
         
        
    def straighten(self,Pos, P_spine,rbins, Vel=None,sort=None, Num_cum=None, Num_W=None ):
        """straightens the curved filament by assuming that the filament spine is a piecewise continuous curve, 
        created by straight line segments joining consecutive points in the discretely sampled spine.
        All the partilces in cylinders with rmin<r<rmax, and axes along these segments are rotated and translated so as to 
        lie besides each other, forming a straight cylinderical filament.
        
        Inputs:
        Pos: positions of particles of the box, in the shape [Np, 3]  (sorted, in case sort is not None)
        P_spine: discretely sampled points on the spine, in the shape [Ns, 3]
        rbins: radial bins
        Vel: velocities of the particles in the shape [Np, 3] (sorted, in case sort is not None)
        sort, Num_cum, Num_W: information about the sorted particles 
        """
        filx=P_spine[:,0]
        Pos_straight=[]
        Vel_straight=[]
        z_start=0
        B=self.B
        
        if (sort is None):
            for i in range(len(filx)-1):
                fil1=P_spine[i] #first end point of the segment
                fil2=P_spine[i+1] #second end point of the segment
                if(self.pbc is True):
                    fil2=np.subtract(fil2, fil1-l/2)%l-l/2+fil1 #accounting for PBC, unfolding

                r_sep=fil2-fil1
                dc=r_sep/np.linalg.norm(r_sep) #direction cosines of the filament segment
                
                Pos2=Pos.copy() 
                if (Vel is not None):
                    Vel2=Vel.copy()
                    
                #translating to the frame of reference of the origin of the segment
                if(self.pbc is True):
                    Pos2=(np.subtract(np.subtract(Pos2, (fil1-l/2))%l, l/2))
                    fil2=(np.subtract(np.subtract(fil2, (fil1-l/2))%l, l/2))
                    fil1=(np.subtract(np.subtract(fil1, (fil1-l/2))%l, l/2))
                else:
                    Pos2=np.subtract(Pos2, fil1)
                    fil2=np.subtract(fil2, fil1)
                    fil1=np.subtract(fil1, fil1)


                #rotating so that the filament is along the z axis
                Pos_rot=self.rotate(np.transpose(Pos2), r_sep, B)
                r1_rot=self.rotate(fil1, r_sep, B)
                r2_rot=self.rotate(fil2, r_sep, B)
                
                Pos_rot=Pos_rot.T
                if (Vel is not None):
                    Vel_rot=self.rotate(np.transpose(Vel2), r_sep, B)
                    Vel_rot=Vel_rot.T

                #now, the segment is along the z axis
                minz=0 #maximum z of the segment
                maxz=r2_rot[2] #minimum z of the segment    

                  
                r_perp=np.sqrt((Pos_rot[:,0])**2+(Pos_rot[:,1])**2) #distance perpendicular to the spine
                ind1=np.where((Pos_rot[:,2]>=minz)&(Pos_rot[:,2]<=maxz)&(r_perp>=rbins[0])&(r_perp<=rbins[-1]))[0]
                Pos_rot=Pos_rot[ind1]
                Pos_rot[:,2]+=z_start
                Pos_straight.append(Pos_rot)
                z_start+=r2_rot[2] #the next filament will start at this z value.
                
                if (Vel is not None):
                    Vel_rot=Vel_rot[ind1]
                    Vel_straight.append(Vel_rot)
                    
        # if sort is not none
        else:
            for i in range(len(filx)-1):
                fil1=P_spine[i] #first end point of the segment
                fil2=P_spine[i+1] #second end point of the segment
                if(self.pbc is True):
                    fil2=np.subtract(fil2, fil1-l/2)%l-l/2+fil1 #accounting for PBC, unfolding
                        
                r_sep=fil2-fil1
                dc=r_sep/np.linalg.norm(r_sep) #direction cosines of the filament segment
                ###############################################################################################
                #indices to search for
                maxr=np.array([max(fil2[0], fil1[0]), max(fil2[1], fil1[1]),  max(fil2[2], fil1[2])])
                minr=np.array([min(fil2[0], fil1[0]), min(fil2[1], fil1[1]),  min(fil2[2], fil1[2])])

                maxr=maxr+np.max(rbins)
                minr=minr-np.max(rbins)

                #floor is required to round down negative values   
                indmax=np.array(np.floor(maxr/self.l), dtype=int) #maximum index of sub-boxes along each direction
                indmin=np.array(np.floor(minr/self.l), dtype=int)

                xind=np.array(np.arange(indmin[0], indmax[0]+1,1), dtype=int)
                yind=np.array(np.arange(indmin[1], indmax[1]+1,1), dtype=int)
                zind=np.array(np.arange(indmin[2], indmax[2]+1,1), dtype=int)

                X, Y, Z= np.meshgrid(xind, yind, zind)
                X=X.flatten()
                Y=Y.flatten()
                Z=Z.flatten()
                weights=W(X, Y, Z, nbox) #sub-boxes to search in this takes into account PBC

                X=Y=Z=0 #clear space

                weight0=weights[0] #first sub-box
                Pos2=Pos[Num_cum[weight0]:Num_cum[weight0]+Num_W[weight0],:]
                if (Vel is not None):
                    Vel2=Vel[Num_cum[weight0]:Num_cum[weight0]+Num_W[weight0],:]

                for b in range(1,len(weights),1):
                    Pos1=Pos[Num_cum[weights[b]]:Num_cum[weights[b]]+Num_W[weights[b]],:]
                    Pos2=np.concatenate([Pos2,Pos1],axis=0)
                    
                    if(Vel is not None):
                        Vel1=Vel[Num_cum[weights[b]]:Num_cum[weights[b]]+Num_W[weights[b]],:]
                        Vel2=np.concatenate([Vel2,Vel1],axis=0)      
                #####################################################################################################
                 #translating to the frame of reference of the origin of the segment
                if (self.pbc is True):
                    Pos2=(np.subtract(np.subtract(Pos2, (fil1-l/2))%l, l/2))
                    fil2=(np.subtract(np.subtract(fil2, (fil1-l/2))%l, l/2)) #make sure that fil2 comes before fil1 is modified
                    fil1=(np.subtract(np.subtract(fil1, (fil1-l/2))%l, l/2))
                else:
                    Pos2=np.subtract(Pos2, fil1)
                    fil2=np.subtract(fil2, fil1)
                    fil1=np.subtract(fil1, fil1)


                #rotating so that the filament is along the z axis
                Pos_rot=self.rotate(np.transpose(Pos2), r_sep, B)
                r1_rot=self.rotate(fil1, r_sep, B)
                r2_rot=self.rotate(fil2, r_sep, B)
                
                Pos_rot=Pos_rot.T
                if (Vel is not None):
                    Vel_rot=self.rotate(np.transpose(Vel2), r_sep, B)
                    Vel_rot=Vel_rot.T

                #now, the segment is along the z axis
                minz=0 #maximum z of the segment
                maxz=r2_rot[2] #minimum z of the segment    

                  
                r_perp=np.sqrt((Pos_rot[:,0])**2+(Pos_rot[:,1])**2) #distance perpendicular to the spine
                ind1=np.where((Pos_rot[:,2]>=minz)&(Pos_rot[:,2]<=maxz)&(r_perp>=rbins[0])&(r_perp<=rbins[-1]))[0]
                Pos_rot=Pos_rot[ind1]
                Pos_rot[:,2]+=z_start
                Pos_straight.append(Pos_rot)
                z_start+=r2_rot[2] #the next filament will start at this z value.
                
                if (Vel is not None):
                    Vel_rot=Vel_rot[ind1]
                    Vel_straight.append(Vel_rot)


        #############################################################################################################
        Pos_straight=np.concatenate(Pos_straight, axis=0)
        if (Vel is not None):
            Vel_straight=np.concatenate(Vel_straight, axis=0)
            
        elif (Vel is None):
            Vel_straight=0
            
        D=dict()
        D["Pos_straight"]=Pos_straight
        D["Vel_straight"]=Vel_straight
        D["lfil"]=z_start

        return D
    
    def profile(self,Pos_straight, rmin, rmax, lfil, Vel_straight=None, dlogr=None, dr=None, N_r=None, N_z=None, log=True):
        """ Calculates the 2d, logitudinal and radial profiles for the given input.
        
        Inputs:
        Pos_straight: positions of particles in the box, aligned such that the spine of the filament is along the z-axis
                        shape is [Np, 3]
        rmin, rmax: bounds on radial bins
        lfil: length of the filament
        Vel_straight: aligned velocities, if present.
        dlogr (dr): radial spacing for log (linear) radial bins
        N_r: number of radial bins
        N_z: number of logitudian bins
        log: True for logarithmic bins in r, False for linear. Default is True.
        
        outputs:
        dictionary D with keys:
        "rho_2d": 2D density profile, shape [N_r, N_z]
        "rho_r": logitudinally averaged radial profile
        "rho_z": radially averaged longitudinal profile
        
        similarly for vz, vr, sigvz, sigvr in case Vel_straight is not None.
        rbins, rmid, zbins, zmid: end points and mid points of radial and longitudinal bins respectively"""
                
        ZB=self.create_zbins(lfil, N_z)
        zbins=ZB["zbins"]
        zmid=ZB["zmid"]
        
        RB=self.create_rbins(rmin, rmax, dlogr, dr, N_r, log)
        rbins=RB["rbins"]
        rmid=RB["rmid"]
        ##########################################################################################
        #defining arrays to store outputs
        # 2d profiles
        rho=np.zeros([len(rmid), N_z])
        Num=rho.copy()
        Vol=rho.copy()
        vz=rho.copy()
        vr=rho.copy()
        sigvz=rho.copy()
        sigvr=rho.copy()

        # 1d profiles
        rho_r=np.zeros(len(rmid)) # radial density profile
        rho_z=np.zeros(N_z) # tangential density profile

        Vol_r=rho_r.copy()
        Vol_z=rho_z.copy()

        vz_r=rho_r.copy()
        vz_z=rho_z.copy()
        vr_r=rho_r.copy()
        vr_z=rho_z.copy()

        sigvz_r=rho_r.copy()
        sigvz_z=rho_z.copy()
        sigvr_r=rho_r.copy()
        sigvr_z=rho_z.copy()
        ################################################################################################
        # removing particles outside the cylinder of interest
        ind=np.where((Pos_straight[:,2]>=zbins[0])+(Pos_straight[:,2]<zbins[-1]))[0]
        Pos_straight=Pos_straight[ind]
        # r in cylindrical coordinates
        r_perp=np.sqrt(Pos_straight[:,0]**2+Pos_straight[:,1]**2)
        
        if (Vel_straight is not None):
            Vel_straight=Vel_straight[ind]
            Velr=(Vel_straight[:,0]*Pos_straight[:,0]+Vel_straight[:,1]*Pos_straight[:,1])/r_perp #radial velocity
        
        ind=np.where((r_perp>=rbins[0])&(r_perp<rbins[-1]))[0]
        Pos_straight=Pos_straight[ind]
        r_perp=r_perp[ind]
        
        if (Vel_straight is not None):
            Vel_straight=Vel_straight[ind]
            Velr=Velr[ind]
        
        # calculating the profile
        for u in range(len(rmid)):
            ind=np.where((r_perp>=rbins[u])&(r_perp<rbins[u+1]))[0]
            Pos1=Pos_straight[ind]
            vol=lfil*np.pi*((rbins[u+1])**2-(rbins[u])**2)
            Vol_r[u]=vol
            rho_r[u]=len(Pos1[:,0])/vol
            
            if ((Vel_straight is not None)&(len(Pos1[:,0])!=0)):
                Vel1=Vel_straight[ind]
                Velr1=Velr[ind]

                vz_r[u]=np.mean(Vel1[:,2])
                sigvz_r[u]=np.std(Vel1[:,2])
                vr_r[u]=np.mean(Velr1)
                sigvr_r[u]=np.std(Velr1)

            #looping over zbins
            for q in range(N_z):
                ind3=np.where((Pos1[:,2]>=zbins[q])&(Pos1[:,2]<zbins[q+1]))[0]
                Pos2=Pos1[ind3]
                vol=np.pi*(rbins[u+1]**2-rbins[u]**2)*(zbins[q+1]-zbins[q])
                rho[u][q]=len(Pos2[:,0])/vol
                Num[u][q]=len(Pos2[:,0])
                Vol[u][q]=vol
                
                if ((Vel_straight is not None)&(len(Pos2[:,0])!=0)):
                    Vel2=Vel1[ind3]
                    Velr2=Velr1[ind3]
                    
                    vz[u][q]=np.mean(Vel2[:,2])
                    sigvz[u][q]=np.std(Vel2[:,2])
                    vr[u][q]=np.mean(Velr2)
                    sigvr[u][q]=np.std(Velr2)
                    

        for t in range(N_z):
            ind2=np.where((Pos_straight[:,2]>=zbins[t])&(Pos_straight[:,2]<zbins[t+1]))[0]
            Pos1=Pos_straight[ind2]
            vol=(zbins[t+1]-zbins[t])*np.pi*((rbins[-1])**2-rbins[0]**2)
            Vol_z[t]=vol
            rho_z[t]=len(Pos1[:,0])/vol
            
            if ((Vel_straight is not None)&(len(Pos1[:,0])!=0)):
                Vel1=Vel_straight[ind2]
                Velr1=Velr[ind2]

                vz_z[t]=np.mean(Vel1[:,2])
                sigvz_z[t]=np.std(Vel1[:,2])
                vr_z[t]=np.mean(Velr1)
                sigvr_z[t]=np.std(Velr1)


        rho=rho*self.mp
        rho_z=rho_z*self.mp
        rho_r=rho_r*self.mp
        
        D=dict()
        D["rho"]=rho
        D["Num"]=Num
        D["Vol"]=Vol
        D["vz"]=vz
        D["vr"]=vr
        D["sigvr"]=sigvr
        D["sigvz"]=sigvz
        
        D["rho_r"]=rho_r
        D["rho_z"]=rho_z
        D["Vol_r"]=Vol_r
        D["Vol_z"]=Vol_z
        D["vz_r"]=vz_r
        D["vz_z"]=vz_z
        D["vr_r"]=vr_r
        D["vr_z"]=vr_z
        
        D["sigvz_r"]=sigvz_r
        D["sigvz_z"]=sigvz_z
        D["sigvr_r"]=sigvr_r
        D["sigvr_z"]=sigvr_z
        
        D["rbins"]=rbins
        D["rmid"]=rmid
        D["zbins"]=zbins
        D["zmid"]=zmid
        return D
    
    ########################################################################################################  
    
    def estimate_profile(self,Pos, P_spine, rmin, rmax, dlogr=None, dr=None, N_r=None, N_z=None, log=True, Vel=None,
                         sort=None, Num_cum=None, Num_W=None):
        """ estimates the 2d and 1d profiles of filaments.
        Inputs:
        Pos: positions of particles of the box, in the shape [Np, 3]  (sorted, in case sort is not None)
        P_spine: discretely sampled points on the spine, in the shape [Ns, 3]
        rmin, rmax: inner and outer radius of the cylinder
        dlogr (dr): radial spacing for log (linear) radial bins
        N_r: number of radial bins
        N_z: number of logitudian bins
        log: True for logarithmic bins in r, False for linear. Default is True.
        
        Vel: velocities of the particles in the shape [Np, 3] (sorted, in case sort is not None)
        sort, Num_cum, Num_W: information about the sorted particles 
        
        Outputs:
        dictionary D with following keys:
        2d profiles: rho, vz, vr , sigvz, sigvr
        1d profiles: above quantities {q} with 
        q_r: logitudinal averaged radial profiles
        q_z: radial averaged logitudinal profiles
        rbins, rmid, zbins, zmid: end points and mid points of radial and longitudinal bins respectively"""
        
        D=self.create_rbins(rmin, rmax, dlogr, dr, N_r, log)
        rbins=D["rbins"]

        D1=self.straighten(Pos, P_spine, rbins, Vel,sort, Num_cum, Num_W)
        Pos_straight=D1["Pos_straight"]
        lfil=D1["lfil"]
        if (Vel is not None):
            Vel_straight=D1["Vel_straight"]
        else:
            Vel_straight=None
         
        D=self.profile(Pos_straight, rmin, rmax, lfil, Vel_straight, dlogr, dr, N_r, N_z, log)
        return D
    ########################################################################################################
    def profile_3d(self,Pos_straight, rmin, rmax, lfil, Vel_straight=None, dlogr=None, dr=None, 
                    N_r=None, N_z=None, N_phi=20, log=True):              
            
        """ Calculates the 3d, logitudinal and radial profiles for the given input.
        
        Inputs:
        Pos_straight: positions of particles in the box, aligned such that the spine of the filament is along the z-axis
                        shape is [Np, 3]
        rmin, rmax: bounds on radial bins
        lfil: length of the filament
        Vel_straight: aligned velocities, if present.
        dlogr (dr): radial spacing for log (linear) radial bins
        N_r: number of radial bins
        N_z: number of logitudian bins
        N_phi: number of phi bins
        log: True for logarithmic bins in r, False for linear. Default is True.
        
        outputs:
        dictionary D with keys:
        "rho_3d": 3D density profile, shape [N_r, N_z, N_phi]
        "rho_r": logitudinally averaged radial profile
        "rho_z": radially averaged longitudinal profile
        
        similarly for vz, vr, sigvz, sigvr in case Vel_straight is not None.
        rbins, rmid, zbins, zmid, phibins, phimid: 
        end points and mid points of radial, longitudinal, and azimuthal bins respectively"""
                
        ZB=self.create_zbins(lfil, N_z)
        zbins=ZB["zbins"]
        zmid=ZB["zmid"]
        
        RB=self.create_rbins(rmin, rmax, dlogr, dr, N_r, log)
        rbins=RB["rbins"]
        rmid=RB["rmid"]
        
        phibins=np.linspace(0, 2*np.pi,N_phi+1) 
        phimid=np.zeros(N_phi)
        for p in range(N_phi):
            phimid[p]=(phibins[p]+phibins[p+1])/2
            
        ##########################################################################################
        #defining arrays to store outputs
        # 3d profiles
        rho=np.zeros([len(rmid), N_z, N_phi])
        Num=rho.copy()
        Vol=rho.copy()
        vz=rho.copy()
        vr=rho.copy()
        sigvz=rho.copy()
        sigvr=rho.copy()

        # 1d profiles
        rho_r=np.zeros(len(rmid)) # radial density profile
        rho_z=np.zeros(N_z) # tangential density profile

        Vol_r=rho_r.copy()
        Vol_z=rho_z.copy()

        vz_r=rho_r.copy()
        vz_z=rho_z.copy()
        vr_r=rho_r.copy()
        vr_z=rho_z.copy()

        sigvz_r=rho_r.copy()
        sigvz_z=rho_z.copy()
        sigvr_r=rho_r.copy()
        sigvr_z=rho_z.copy()
        ################################################################################################
        # removing particles outside the cylinder of interest
        ind=np.where((Pos_straight[:,2]>=zbins[0])+(Pos_straight[:,2]<zbins[-1]))[0]
        Pos_straight=Pos_straight[ind]
        # r in cylindrical coordinates
        r_perp=np.sqrt(Pos_straight[:,0]**2+Pos_straight[:,1]**2)
        phi=np.arctan2(Pos_straight[:,1], Pos_straight[:,0])
        phi[phi<0]=phi[phi<0]+np.pi*2 # phi values from 0 to 2pi
        
        if (Vel_straight is not None):
            Vel_straight=Vel_straight[ind]
            Velr=(Vel_straight[:,0]*Pos_straight[:,0]+Vel_straight[:,1]*Pos_straight[:,1])/r_perp #radial velocity
        
        ind=np.where((r_perp>=rbins[0])&(r_perp<rbins[-1]))[0]
        Pos_straight=Pos_straight[ind]
        r_perp=r_perp[ind]
        
        if (Vel_straight is not None):
            Vel_straight=Vel_straight[ind]
            Velr=Velr[ind]
        
        # calculating the profile
        for u in range(len(rmid)):
            ind=np.where((r_perp>=rbins[u])&(r_perp<rbins[u+1]))[0]
            Pos1=Pos_straight[ind]
            phi1=phi[ind]
            
            vol=lfil*np.pi*((rbins[u+1])**2-(rbins[u])**2)
            Vol_r[u]=vol
            rho_r[u]=len(Pos1[:,0])/vol
            
            if ((Vel_straight is not None)&(len(Pos1[:,0])!=0)):
                Vel1=Vel_straight[ind]
                Velr1=Velr[ind]

                vz_r[u]=np.mean(Vel1[:,2])
                sigvz_r[u]=np.std(Vel1[:,2])
                vr_r[u]=np.mean(Velr1)
                sigvr_r[u]=np.std(Velr1)

            #looping over zbins
            for q in range(N_z):
                ind3=np.where((Pos1[:,2]>=zbins[q])&(Pos1[:,2]<zbins[q+1]))[0]
                Pos2=Pos1[ind3]
                phi2=phi1[ind3]
                if ((Vel_straight is not None)&(len(Pos2[:,0])!=0)):
                        Vel2=Vel1[ind3]
                        Velr2=Velr1[ind3]
                for ph in range(N_phi):
                    ind4=np.where((phi2>=phibins[ph])&(phi2<phibins[ph+1]))[0]
                    Pos3=Pos2[ind4]
                    
                    vol=np.pi*(rbins[u+1]**2-rbins[u]**2)*(zbins[q+1]-zbins[q])*(phibins[ph+1]-phibins[ph])/(2*np.pi)
                    rho[u][q][ph]=len(Pos3[:,0])/vol
                    Num[u][q][ph]=len(Pos3[:,0])
                    Vol[u][q][ph]=vol

                    if ((Vel_straight is not None)&(len(Pos3[:,0])!=0)):
                        Vel3=Vel2[ind4]
                        Velr3=Velr2[ind4]
                  
                        vz[u][q][ph]=np.mean(Vel3[:,2])
                        sigvz[u][q][ph]=np.std(Vel3[:,2])
                        vr[u][q][ph]=np.mean(Velr3)
                        sigvr[u][q][ph]=np.std(Velr3)


        for t in range(N_z):
            ind2=np.where((Pos_straight[:,2]>=zbins[t])&(Pos_straight[:,2]<zbins[t+1]))[0]
            Pos1=Pos_straight[ind2]
            vol=(zbins[t+1]-zbins[t])*np.pi*((rbins[-1])**2-rbins[0]**2)
            Vol_z[t]=vol
            rho_z[t]=len(Pos1[:,0])/vol
            
            if ((Vel_straight is not None)&(len(Pos1[:,0])!=0)):
                Vel1=Vel_straight[ind2]
                Velr1=Velr[ind2]

                vz_z[t]=np.mean(Vel1[:,2])
                sigvz_z[t]=np.std(Vel1[:,2])
                vr_z[t]=np.mean(Velr1)
                sigvr_z[t]=np.std(Velr1)


        rho=rho*self.mp
        rho_z=rho_z*self.mp
        rho_r=rho_r*self.mp
        
        D=dict()
        D["rho"]=rho
        D["Num"]=Num
        D["Vol"]=Vol
        D["vz"]=vz
        D["vr"]=vr
        D["sigvr"]=sigvr
        D["sigvz"]=sigvz
        
        D["rho_r"]=rho_r
        D["rho_z"]=rho_z
        D["Vol_r"]=Vol_r
        D["Vol_z"]=Vol_z
        D["vz_r"]=vz_r
        D["vz_z"]=vz_z
        D["vr_r"]=vr_r
        D["vr_z"]=vr_z
        
        D["sigvz_r"]=sigvz_r
        D["sigvz_z"]=sigvz_z
        D["sigvr_r"]=sigvr_r
        D["sigvr_z"]=sigvr_z
        
        D["rbins"]=rbins
        D["rmid"]=rmid
        D["zbins"]=zbins
        D["zmid"]=zmid
        return D
    
    ########################################################################################################  
    
    def estimate_profile_3d(self,Pos, P_spine, rmin, rmax, dlogr=None, dr=None, 
                            N_r=None, N_z=None,N_phi=20, log=True, Vel=None,
                         sort=None, Num_cum=None, Num_W=None):
        """ estimates the 3d and 1d profiles of filaments.
        Inputs:
        Pos: positions of particles of the box, in the shape [Np, 3]  (sorted, in case sort is not None)
        P_spine: discretely sampled points on the spine, in the shape [Ns, 3]
        rmin, rmax: inner and outer radius of the cylinder
        dlogr (dr): radial spacing for log (linear) radial bins
        N_r: number of radial bins
        N_z: number of logitudian bins
        N_phi: number of phi bins
        log: True for logarithmic bins in r, False for linear. Default is True.
        
        Vel: velocities of the particles in the shape [Np, 3] (sorted, in case sort is not None)
        sort, Num_cum, Num_W: information about the sorted particles 
        
        Outputs:
        dictionary D with following keys:
        3d profiles: rho, vz, vr , sigvz, sigvr
        1d profiles: above quantities {q} with 
        q_r: logitudinal averaged radial profiles
        q_z: radial averaged logitudinal profiles
        rbins, rmid, zbins, zmid: end points and mid points of radial and longitudinal bins respectively"""
        
        D=self.create_rbins(rmin, rmax, dlogr, dr, N_r, log)
        rbins=D["rbins"]
        phibins=np.linspace(0, 2*np.pi,N_phi+1) 
        
        D1=self.straighten(Pos, P_spine, rbins, Vel,sort, Num_cum, Num_W)
        Pos_straight=D1["Pos_straight"]
        lfil=D1["lfil"]
        if (Vel is not None):
            Vel_straight=D1["Vel_straight"]
        else:
            Vel_straight=None
         
        D=self.profile_3d(Pos_straight, rmin, rmax, lfil, Vel_straight, dlogr, dr, N_r, N_z,N_phi, log)
        return D
    
    
    
    # curvature info

    ########################################################################################################  
    
    def estimate_profile_3d_curve(self,Pos, P_spine, rmin, rmax, dlogr=None, dr=None, 
                            N_r=None, N_z=None,N_phi=20, log=True, nk=3, Vel=None,
                         sort=None, Num_cum=None, Num_W=None):
        """ estimates the 3d and 1d profiles of filaments, split by curvature into nk types.
        Inputs:
        Pos: positions of particles of the box, in the shape [Np, 3]  (sorted, in case sort is not None)
        P_spine: discretely sampled points on the spine, in the shape [Ns, 3]
        rmin, rmax: inner and outer radius of the cylinder
        dlogr (dr): radial spacing for log (linear) radial bins
        N_r: number of radial bins
        N_z: number of logitudian bins
        N_phi: number of phi bins
        log: True for logarithmic bins in r, False for linear. Default is True.
        nk: number of parts the filaments is to be split into based on curvture.
        
        Vel: velocities of the particles in the shape [Np, 3] (sorted, in case sort is not None)
        sort, Num_cum, Num_W: information about the sorted particles 
        
        Outputs:
        dictionary D with following keys:
        3d profiles: rho, vz, vr , sigvz, sigvr
        1d profiles: above quantities {q} with 
        q_r: logitudinal averaged radial profiles
        q_z: radial averaged logitudinal profiles
        rbins, rmid, zbins, zmid: end points and mid points of radial and longitudinal bins respectively"""
        
        D=self.create_rbins(rmin, rmax, dlogr, dr, N_r, log)
        rbins=D["rbins"]
        phibins=np.linspace(0, 2*np.pi,N_phi+1) 
        
        # calculating curvature
        kappa_l=np.zeros(len(P_spine)-1)
        kappa_h=np.zeros(len(P_spine)-1)
        for i in range(1,len(kappa_l)-1,1):
            v0=P_spine[i]-P_spine[i-1]
            v1=P_spine[i+1]-P_spine[i]
            v2=P_spine[i+2]-P_spine[i+1]
            l0=np.linalg.norm(v0)
            l1=np.linalg.norm(v1)
            l2=np.linalg.norm(v2)

            mu_h=np.dot(v1, v2)/(l1*l2)
            mu_l=np.dot(v0, v1)/(l0*l1)
            if (mu_h>1):
                mu_h=1
            elif (mu_h<-1):
                mu_h=-1
            theta_h=np.arccos(mu_h)

            if (mu_l>1):
                mu_l=1
            elif (mu_l<-1):
                mu_l=-1
            theta_l=np.arccos(mu_l)

            kappa_l[i]=theta_l/l1
            kappa_h[i]=theta_h/l1


        # for the first segment
        v1=P_spine[1]-P_spine[0]
        v2=P_spine[2]-P_spine[1]
        l1=np.linalg.norm(v1)
        l2=np.linalg.norm(v2)

        mu_h=np.dot(v1, v2)/(l1*l2)
        if (mu_h>1):
            mu_h=1
        elif (mu_h<-1):
            mu_h=-1
        theta_h=np.arccos(mu_h)
        kappa_h[0]=theta_h/l1

        # for the last segment
        v1=P_spine[len(kappa_l)]-P_spine[len(kappa_l)-1]
        v0=P_spine[len(kappa_l)-1]-P_spine[len(kappa_l)-2]
        l1=np.linalg.norm(v1)
        l0=np.linalg.norm(v0)

        mu_l=np.dot(v1, v0)/(l1*l0)
        if (mu_l>1):
            mu_l=1
        elif (mu_l<-1):
            mu_l=-1
        theta_l=np.arccos(mu_l)
        kappa_l[-1]=theta_l/l1
        
        # averaging
        kappa_tot=(kappa_l+kappa_h)/2
        kappa_tot[0]=kappa_h[0]
        kappa_tot[-1]=kappa_l[-1]
        
        k1, k2=np.percentile(kappa_tot, [33, 66])
        low=[]
        mid=[]
        high=[]
        for i in range(len(kappa_tot)):
            if (kappa_tot[i]<k1):
                low.append(i)
            elif ((kappa_tot[i]>=k1)&(kappa_tot[i]<k2)):
                mid.append(i)
            else:
                high.append(i)
        
        
        #now, straightening, but splitting
        filx=P_spine[:,0]
        Pos_straight_low=[]
        Vel_straight_low=[]
        Pos_straight_mid=[]
        Vel_straight_mid=[]
        Pos_straight_high=[]
        Vel_straight_high=[]
        
        Pos_straight=[]
        Vel_straight=[]
        z_start=0
        B=self.B
        
        if (sort is None):
            for i in range(len(filx)-1):
                fil1=P_spine[i] #first end point of the segment
                fil2=P_spine[i+1] #second end point of the segment
                if(self.pbc is True):
                    fil2=np.subtract(fil2, fil1-l/2)%l-l/2+fil1 #accounting for PBC, unfolding

                r_sep=fil2-fil1
                dc=r_sep/np.linalg.norm(r_sep) #direction cosines of the filament segment
                
                Pos2=Pos.copy() 
                if (Vel is not None):
                    Vel2=Vel.copy()
                    
                #translating to the frame of reference of the origin of the segment
                if(self.pbc is True):
                    Pos2=(np.subtract(np.subtract(Pos2, (fil1-l/2))%l, l/2))
                    fil2=(np.subtract(np.subtract(fil2, (fil1-l/2))%l, l/2))
                    fil1=(np.subtract(np.subtract(fil1, (fil1-l/2))%l, l/2))
                else:
                    Pos2=np.subtract(Pos2, fil1)
                    fil2=np.subtract(fil2, fil1)
                    fil1=np.subtract(fil1, fil1)


                #rotating so that the filament is along the z axis
                Pos_rot=self.rotate(np.transpose(Pos2), r_sep, B)
                r1_rot=self.rotate(fil1, r_sep, B)
                r2_rot=self.rotate(fil2, r_sep, B)
                
                Pos_rot=Pos_rot.T
                if (Vel is not None):
                    Vel_rot=self.rotate(np.transpose(Vel2), r_sep, B)
                    Vel_rot=Vel_rot.T

                #now, the segment is along the z axis
                minz=0 #maximum z of the segment
                maxz=r2_rot[2] #minimum z of the segment    

                  
                r_perp=np.sqrt((Pos_rot[:,0])**2+(Pos_rot[:,1])**2) #distance perpendicular to the spine
                ind1=np.where((Pos_rot[:,2]>=minz)&(Pos_rot[:,2]<=maxz)&(r_perp>=rbins[0])&(r_perp<=rbins[-1]))[0]
                Pos_rot=Pos_rot[ind1]
                Pos_rot[:,2]+=z_start
                Pos_straight.append(Pos_rot)
                z_start+=r2_rot[2] #the next filament will start at this z value.
                
                if i in low:
                    Pos_straight_low.append(Pos_rot)
                elif i in mid:
                    Pos_straight_mid.append(Pos_rot)
                else:
                    Pos_straight_high.append(Pos_rot)
                    
                if (Vel is not None):
                    Vel_rot=Vel_rot[ind1]
                    Vel_straight.append(Vel_rot)
                    
                    if i in low:
                        Vel_straight_low.append(Vel_rot)
                    elif i in mid:
                        Vel_straight_mid.append(Vel_rot)
                    else:
                        Vel_straight_high.append(Vel_rot)
                    
        # if sort is not none
        else:
            for i in range(len(filx)-1):
                fil1=P_spine[i] #first end point of the segment
                fil2=P_spine[i+1] #second end point of the segment
                if(self.pbc is True):
                    fil2=np.subtract(fil2, fil1-l/2)%l-l/2+fil1 #accounting for PBC, unfolding
                        
                r_sep=fil2-fil1
                dc=r_sep/np.linalg.norm(r_sep) #direction cosines of the filament segment
                ###############################################################################################
                #indices to search for
                maxr=np.array([max(fil2[0], fil1[0]), max(fil2[1], fil1[1]),  max(fil2[2], fil1[2])])
                minr=np.array([min(fil2[0], fil1[0]), min(fil2[1], fil1[1]),  min(fil2[2], fil1[2])])

                maxr=maxr+np.max(rbins)
                minr=minr-np.max(rbins)

                #floor is required to round down negative values   
                indmax=np.array(np.floor(maxr/self.l), dtype=int) #maximum index of sub-boxes along each direction
                indmin=np.array(np.floor(minr/self.l), dtype=int)

                xind=np.array(np.arange(indmin[0], indmax[0]+1,1), dtype=int)
                yind=np.array(np.arange(indmin[1], indmax[1]+1,1), dtype=int)
                zind=np.array(np.arange(indmin[2], indmax[2]+1,1), dtype=int)

                X, Y, Z= np.meshgrid(xind, yind, zind)
                X=X.flatten()
                Y=Y.flatten()
                Z=Z.flatten()
                weights=W(X, Y, Z, nbox) #sub-boxes to search in this takes into account PBC

                X=Y=Z=0 #clear space

                weight0=weights[0] #first sub-box
                Pos2=Pos[Num_cum[weight0]:Num_cum[weight0]+Num_W[weight0],:]
                if (Vel is not None):
                    Vel2=Vel[Num_cum[weight0]:Num_cum[weight0]+Num_W[weight0],:]

                for b in range(1,len(weights),1):
                    Pos1=Pos[Num_cum[weights[b]]:Num_cum[weights[b]]+Num_W[weights[b]],:]
                    Pos2=np.concatenate([Pos2,Pos1],axis=0)
                    
                    if(Vel is not None):
                        Vel1=Vel[Num_cum[weights[b]]:Num_cum[weights[b]]+Num_W[weights[b]],:]
                        Vel2=np.concatenate([Vel2,Vel1],axis=0)      
                #####################################################################################################
                 #translating to the frame of reference of the origin of the segment
                if (self.pbc is True):
                    Pos2=(np.subtract(np.subtract(Pos2, (fil1-l/2))%l, l/2))
                    fil2=(np.subtract(np.subtract(fil2, (fil1-l/2))%l, l/2)) #make sure that fil2 comes before fil1 is modified
                    fil1=(np.subtract(np.subtract(fil1, (fil1-l/2))%l, l/2))
                else:
                    Pos2=np.subtract(Pos2, fil1)
                    fil2=np.subtract(fil2, fil1)
                    fil1=np.subtract(fil1, fil1)


                #rotating so that the filament is along the z axis
                Pos_rot=self.rotate(np.transpose(Pos2), r_sep, B)
                r1_rot=self.rotate(fil1, r_sep, B)
                r2_rot=self.rotate(fil2, r_sep, B)
                
                Pos_rot=Pos_rot.T
                if (Vel is not None):
                    Vel_rot=self.rotate(np.transpose(Vel2), r_sep, B)
                    Vel_rot=Vel_rot.T

                #now, the segment is along the z axis
                minz=0 #maximum z of the segment
                maxz=r2_rot[2] #minimum z of the segment    

                  
                r_perp=np.sqrt((Pos_rot[:,0])**2+(Pos_rot[:,1])**2) #distance perpendicular to the spine
                ind1=np.where((Pos_rot[:,2]>=minz)&(Pos_rot[:,2]<=maxz)&(r_perp>=rbins[0])&(r_perp<=rbins[-1]))[0]
                Pos_rot=Pos_rot[ind1]
                Pos_rot[:,2]+=z_start
                Pos_straight.append(Pos_rot)
                z_start+=r2_rot[2] #the next filament will start at this z value.
                
                if i in low:
                    Pos_straight_low.append(Pos_rot)
                elif i in mid:
                    Pos_straight_mid.append(Pos_rot)
                else:
                    Pos_straight_high.append(Pos_rot)
                    
                if (Vel is not None):
                    Vel_rot=Vel_rot[ind1]
                    Vel_straight.append(Vel_rot)
                    
                    if i in low:
                        Vel_straight_low.append(Vel_rot)
                    elif i in mid:
                        Vel_straight_mid.append(Vel_rot)
                    else:
                        Vel_straight_high.append(Vel_rot)
                


        #############################################################################################################
        Pos_straight=np.concatenate(Pos_straight, axis=0)
        
        Pos_straight_low=np.concatenate(Pos_straight_low, axis=0)
        Pos_straight_mid=np.concatenate(Pos_straight_mid, axis=0)
        Pos_straight_high=np.concatenate(Pos_straight_high, axis=0)
        if (Vel is not None):
            Vel_straight=np.concatenate(Vel_straight, axis=0)
            Vel_straight_low=np.concatenate(Vel_straight_low, axis=0)
            Vel_straight_mid=np.concatenate(Vel_straight_mid, axis=0)
            Vel_straight_high=np.concatenate(Vel_straight_high, axis=0)
            
        elif (Vel is None):
            Vel_straight=0
            Vel_straight_low=0
            Vel_straight_mid=0
            Vel_straight_high=0
            
        lfil=z_start
        # there's a need to renormalise the densities obtained
         
        D=self.profile_3d(Pos_straight, rmin, rmax, lfil, Vel_straight, dlogr, dr, N_r, N_z,N_phi, log)
        Dl=self.profile_3d(Pos_straight_low, rmin, rmax, lfil, Vel_straight_low, dlogr, dr, N_r, N_z,N_phi, log)
        Dm=self.profile_3d(Pos_straight_mid, rmin, rmax, lfil, Vel_straight_mid, dlogr, dr, N_r, N_z,N_phi, log)
        Dh=self.profile_3d(Pos_straight_high, rmin, rmax, lfil, Vel_straight_high, dlogr, dr, N_r, N_z,N_phi, log)
        return D, Dl, Dm, Dh
    
    