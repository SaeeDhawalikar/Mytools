import numpy as np
from scipy.interpolate import interp1d 
import scipy.integrate as integrate
from functools import partial
from scipy.optimize import fsolve



class filament_models(object):
    """ models cylindrical profiles of filaments with given cylindrical profile and NFW halos at the ends"""
    
    def __init__(self, lbox,pbc=True,mp=1, Om=0.3,seed=1):
        #mp is the particle mass
        self.l=lbox
        self.pbc=pbc
        self.mp=mp
        self.Om=Om
        np.random.seed(seed)
        
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
    ####################################################################################################################      
    def NFW_rho(self,x, c):
        """ returns the NFW density profile, where
        x=R/R200c,
        c: concentration (R200c/rs)"""
        h= x*(x+1/c)**2
        f= np.log(1+c)-c/(1+c)
        return (3*h*f)**-1

    def NFW_pdf(self,x, c):
        """ returns the pdf of radius distribution for NFW profile,
        x=R/R200c,
        c: concentration (R200c/rs)"""
        A=(np.log(1+c)+1/(1+c)-1)**(-1)
        A1=A*x/(x+1/c)**2
        return A1

    def NFW_cdf(self,x, c):
        """ returns the cdf of radius distribution in NFW profile, where
        x=R/R200c,
        c: concentration (R200c/rs)"""
        A=(np.log(1+c)+1/(1+c)-1)**(-1)
        A1= np.log(c*x+1)+1/(c*x+1)-1
        return A*A1
    
    def g(self,x):
        """ a function required in NFW profile"""
        y=np.log(1+x)-x/(1+x)
        return y
    
    
    def inte_beta(self,beta, y):
        """ a function to be integrated"""
        z=self.g(y)/(y**(3-2*beta)*(1+y)**2)
        return z

    # for non-truncated NFW, set the upper limit of integration to infinity
    def sig2_num_beta(self,x, c, beta):
        '''returns anisotropic sigma_r^2/ (GM/R) for trucated NFW halo
        M: M200c
        R: R200c'''
        t1=partial(self.inte_beta, beta)
        temp=integrate.quad(t1, c*x, c)
        temp1=c**(2-2*beta)*x**(1-2*beta)*(1+c*x)**2/self.g(c)*temp[0]
        return temp1

     
    def create_NFW(self,c, N=int(1e3),R=1, M=None, r0=np.array([0,0,0], dtype=float), seed=None, vel=False, beta=0):
        """ creates a realisation of truncated NFW profile an array of [x, y, z] coordinates in the shape [N,3]
        The halo is truncated at R_{200crit}
        Inputs:
        c: concentration of the halo (R200c/rs)
        N: number of particles in the halo within virial radius
        R: R200c of the halo
        M: mass of the halo (M200c)
        r0: center of the halo
        seed: random seed
        vel: True if velocity information is required (default False)
        beta: velocity anisotropy (default 0)
        
        Output:
        Dictionary D with keys:
        "pos_halo": array of cartesian coordinates of the particles of the halo, shape [N,3]
        "vel_halo": array of velocity cartesian coordinates of the particles of the halo, shape [N,3]"""
        
        if seed is not None:
            np.random.seed(seed)
            
        if M is not None:
            N=int(M/self.mp)
        
        # uniformly sample mu, phi
        phi=np.random.uniform(0, 2*np.pi,N)
        mu=np.random.uniform(-1, 1,N)
        theta=np.arccos(mu)
        
        x_samp=np.array(np.logspace(-5, 0, int(1e4)))# sampling the r/Rvir for cdf interpolating
        cdf_samp=self.NFW_cdf(x_samp, c)
        x_cdf=interp1d(cdf_samp, x_samp)
        cdf_rand=np.random.uniform(0,1, N)
        r=x_cdf(cdf_rand)*R
        
        x_halo=r*np.sin(theta)*np.cos(phi)+r0[0]
        y_halo=r*np.sin(theta)*np.sin(phi)+r0[1]
        z_halo=r*np.cos(theta)+r0[2]
        pos_halo=np.array([x_halo,y_halo, z_halo]).T
        
        if vel is True:
            sigvr=np.zeros(N)
            for i in range(N):
                sigvr[i]=np.abs(np.sqrt(self.sig2_num_beta(r[i]/R, c, beta)))
                
            v200=1000*R #since R is in Mpc/h, H=h*100km/s/Mpc, v200=10RH becomes this in km/
            sigvr=sigvr*v200
            vr_halo=np.random.normal(0, sigvr)
            vtheta_halo=np.random.normal(0, (1-beta)**0.5*sigvr)
            vphi_halo=np.random.normal(0, (1-beta)**0.5*sigvr)
            
            vx_halo=vr_halo*np.sin(theta)*np.cos(phi)+vtheta_halo*np.cos(theta)*np.cos(phi)-vphi_halo*np.sin(phi)
            vy_halo=vr_halo*np.sin(theta)*np.sin(phi)+vtheta_halo*np.cos(theta)*np.sin(phi)+vphi_halo*np.cos(phi)
            vz_halo=vr_halo*np.cos(theta)-vtheta_halo*np.sin(theta)
            
            vel_halo=np.array([vx_halo, vy_halo, vz_halo]).T
        
        else:
            vel_halo=0
        
        D=dict()
        D["pos_halo"]=pos_halo
        D["vel_halo"]=vel_halo

        return D
    
    def create_halo_outskirts(self,c, cdf_r_2h, 
                              vr_mean_prof_2h=None, vr_std_prof_2h=None,
                            vphi_mean_prof_2h=None, vphi_std_prof_2h=None,
                            vtheta_mean_prof_2h=None, vtheta_std_prof_2h=None, 
                              N=int(1e3),R=1, M=None, r0=np.array([0,0,0], dtype=float), seed=None, vel=False):
        """ creates a realisation of halo outskirts, giving positions and velocities of the particles
        The halo is truncated at R_{200crit}
        Inputs:
        c: concentration of the halo (R200c/rs)
        cdf_r_2h: cdf(r), functional type f(r), normalised
        
        
        Output:
        Dictionary D with keys:
        "pos_halo": array of cartesian coordinates of the particles of the halo outskirts, shape [N,3]
        "vel_halo": array of velocity cartesian coordinates of the particles of the halo outskirts, shape [N,3]
        "R2h": 4R200m"""
        
        if seed is not None:
            np.random.seed(seed)

        # uniformly sample mu, phi
        phi=np.random.uniform(0, 2*np.pi,N)
        mu=np.random.uniform(-1, 1,N)
        theta=np.arccos(mu)
        
        # getting 4R200m from R200c
        def sol(w, c):
            y1=np.log(1+w*c)- w*c/(w*c+1)
            y2=np.log(1+c)-c/(1+c)
            y3=w**3*self.Om

            y4=y1/y2-y3
            return y4

        w1=fsolve(sol, 1.1, args=(c))
        R2h=4*w1*R
        
        xsamp=np.array(np.logspace(np.log10(R), np.log10(R2h), int(1e1)))# sampling the r/Rvir for cdf interpolating
        cdf_samp=cdf_r_2h(xsamp)
        x_cdf=interp1d(cdf_samp.flatten(), xsamp.flatten())
        cdf_rand=np.random.uniform(0,np.max(cdf_samp), N)
        r=x_cdf(cdf_rand)
        
        x_halo=r*np.sin(theta)*np.cos(phi)+r0[0]
        y_halo=r*np.sin(theta)*np.sin(phi)+r0[1]
        z_halo=r*np.cos(theta)+r0[2]
        pos_halo=np.array([x_halo,y_halo, z_halo]).T
        
        if vel is True:
            vr_halo=np.random.normal(vr_mean_prof_2h(r, theta, phi), vr_std_prof_2h(r, theta, phi))
            vtheta_halo=np.random.normal(vtheta_mean_prof_2h(r, theta, phi), vtheta_std_prof_2h(r, theta, phi))
            vphi_halo=np.random.normal(vphi_mean_prof_2h(r, theta, phi), vphi_std_prof_2h(r, theta, phi))

            
            vx_halo=vr_halo*np.sin(theta)*np.cos(phi)+vtheta_halo*np.cos(theta)*np.cos(phi)-vphi_halo*np.sin(phi)
            vy_halo=vr_halo*np.sin(theta)*np.sin(phi)+vtheta_halo*np.cos(theta)*np.sin(phi)+vphi_halo*np.cos(phi)
            vz_halo=vr_halo*np.cos(theta)-vtheta_halo*np.sin(theta)
            
            vel_halo=np.array([vx_halo, vy_halo, vz_halo]).T
        
        else:
            vel_halo=0
        
        D=dict()
        D["pos_2h"]=pos_halo
        D["vel_2h"]=vel_halo
        D["R2h"]=R2h

        return D
    # gaussian radial distribution
    def gauss_rho(self,r,s):
        return (2*np.pi*s**2)**(-1)*np.exp(-r**2/(2*s**2))

    #normalised pdf (\propto r*rho(r) for cylindrical profile)
    def gauss_pdf(self,r,s):
        return (1/s**2)*r*np.exp(-r**2/(2*s**2))

    def gauss_cdf(self,r,s):
        y=1-np.exp(-r**2/(2*s**2))
        return y
    
    ###########################################################################################################
    def create_cylinder(self,lcyl, s, N, seed=None, Nsamp=int(1e5)):
        """ creates a realisation of cylindrical gaussian profile of a cylinder oriented along z direction
        and having the starting point at [l/2, l/2, 0]
        inputs:
        lcyl: length of the cylinder
        s: sigma of the radial gaussian profile
        N: number of particles in the cylinder
        seed: random seed
        Nsamp: number of sampling points while interpolating the gaussian cdf (Default int(1e5))
        
        output:
        pos_cyl: cartesian coordinates of the positions of the parti
            is a 2D array with shape [N, 3]
        """
        
        if seed is not None:
            np.random.seed(seed)

        r0=np.logspace(-5,np.log10(5*s), Nsamp) # values of r to interpolate cdf
        # 5s is taken to be the max, as by this s the CDF for sure approaches 1
        CDF=self.gauss_cdf(r0, s) # array of cdf values for given sample of r0 values
        r_interp=interp1d(CDF, r0) # function providing r for given cdf value
        
        if (s<self.l/5):
            cdf1=np.random.random(N) # uniform random sampling of the CDF
            r=r_interp(cdf1)

            phi=np.random.random(N)*np.pi*2 # uniform random phi
            z=np.random.random(N)*lcyl # uniform random z

            x=r*np.cos(phi)
            y=r*np.sin(phi)

        else:
            # First generate 100*N as many points as required,
            #retain only those which lie in the box, then keep only N such points

            cdf1=np.random.random(100*N) # uniform random sampling of the CDF
            r=r_interp(cdf1)

            phi=np.random.random(100*N)*np.pi*2 # uniform random phi
            z=np.random.random(100*N)*lcyl # uniform random z

            x=r*np.cos(phi)
            y=r*np.sin(phi)

            #assuming the filament to be at the center of the box
            ind=np.where((np.abs(x)<=self.l/2)&(np.abs(y)<=self.l/2))[0]
            if (len(ind)<N):
                print("not enough points, retry")
            else:   
                x=x[ind]
                y=y[ind]
                z=z[ind]

                #retaining only N random points
                x=x[:N]
                y=y[:N]
                z=z[:N]
        x=x+self.l/2
        y=y+self.l/2
        pos_cyl=np.array([x, y, z]).T
        return pos_cyl
   
   # check deletion of particles properly 
    def create_filament(self,pos_spine, tang_spine, Nf,s, start_spine=None, 
                        N1=None, c1=None, R1=None, N2=None, c2=None, R2=None,
                        bg=False, mul_bg=200,Nb=None,seed=None,Nsamp=int(1e5)):
        """Creates a realisation of a curved filament, with cylindrical radial profile, and NFW halos at both the ends,
        when the analytical function defining the spine is known.
        
        Inputs:
        pos_spine: function descrbing the position of the spine as a parametric curve
        tang_spine: derivative of pos_spine
        these functions are functions of the type f(t), which returns [x(t), y(t), z(t)] and [x'(t), y'(t), z'(t)] respectively.
        where t runs from [0,1] 
        
        Nf: number of particles in the filament
        s: standard deviation of the gaussian describing radial profile of the filament
        start_spine: staring point of the spine  (default is [l/2, l/2, l/2])
        
        N1, N2: number of particles in the nodes
        c1, c2: concentrations of the nodes (R200c/rs)
        R1, R2: R200c of the nodes
        (if both R1, R2 are specified, R2 is recalculated to be R2=(N2/N1)**(1/3)*R1), so that both the halos have same 
        enclosed overdensity)
        
        bg: uniform background density if bg=True, otherwise background vacuum
        mul_bg: overdensity of the halo wrt critical density (default is 200)
        Nb: number of particles to be distributed uniformly in the background. This must be given if there are no nodes, and bg=True
        seed: random seed
        Nsamp: number of sampling points while interpolating the gaussian cdf (Default int(1e5))
        
        
        output: cartesian coordinates of the positions of all the particles in the box 
             is a 2D array of shape [N,3]
             
        NOTE: if bg is True, the background particles are added on top of the gaussian filament, so that the 
        radial profile of the filament will be gaussian with a positive offset coming from the uniform background.
        If this is undesirable, use create_filament_bg or create_filament_bg1 instead.
        """
        
        if seed is not None:
            np.random.seed(seed)
        if start_spine is None:
            start_spine=np.array([self.l/2, self.l/2, self.l/2])
         
        # length along the spine
        dt=0.0001
        t=np.arange(0, 1+dt, dt)
        spine=pos_spine(t)
        
        d=np.zeros_like(t)
        d0=0
        for i in range(1,len(d),1):
            d1=np.sqrt((spine[0][i]-spine[0][i-1])**2+(spine[1][i]-spine[1][i-1])**2+(spine[2][i]-spine[2][i-1])**2)
            d0=d0+d1
            d[i]=d0
        # writing t as a function of distance along the spine
        t_inter=interp1d(d, t)
        lcyl=np.max(d) # length of the filament
        #################################################################################
        # creating a straight filament
        pos_cyl=self.create_cylinder(lcyl, s, Nf,seed, Nsamp=Nsamp)
        
        # curving the filament
        # translate, rotate, translate
        B=np.array([0,0,1]) # originally, the filament is along z
        pos_new=pos_cyl-np.array([self.l/2, self.l/2, 0]) # taking the filament to x=y=0
        pos_curved=np.zeros_like(pos_cyl)
        for i in range(len(pos_curved)):
            #translate
            pos1=pos_new[i]-np.array([0,0,pos_cyl[i,2]]) # so that z value is 0 always
            d1=pos_new[i,2] #distance along the spine
            t1=t_inter(d1) #corresponding parameter value
            pos_c=pos_spine(t1) #position on the curved profile spine
            tang_c=tang_spine(t1)
            #rotate
            pos2=self.rotate(pos1, B, tang_c)
            #translate
            pos2=pos2+pos_c
            pos_curved[i]=pos2
        spine=spine.T
        pos_curved=pos_curved+(start_spine-spine[0]) # positioning the filament at the starting point   
        pos_full=pos_curved.copy()
        
        # translating the spine
        spine=spine+start_spine-spine[0]
        
        ####################################################################################
        # creating the NFW halos at the ends
        if ((N1 is not None)&(N2 is not None)):
            print("initial R2:", R2)
            R2=(N2/N1)**(1/3)*R1
            print("fixed R2:", R2)
            
        
        if ((N1 is not None)&(c1 is not None)&(R1 is not None)):
            print("creating first node")
            pos_halo1=self.create_NFW(c1, N=N1,R=R1, r0=spine[0])["pos_halo"]
            
            # deleting filament inside the halo
            ind=np.where(np.linalg.norm(pos_full-spine[0], axis=1)<R1)[0]
            pos_full=np.delete(pos_full, ind, axis=0)
            
            pos_full=np.concatenate([pos_full, pos_halo1], axis=0)
            
        if ((N2 is not None)&(c2 is not None)&(R2 is not None)):
            print("creating second node")
            pos_halo2=self.create_NFW(c2, N=N2,R=R2, r0=spine[-1])["pos_halo"]
            # deleting filament inside the halo
            ind=np.where(np.linalg.norm(pos_full-spine[-1], axis=1)<R2)[0]
            pos_full=np.delete(pos_full, ind, axis=0)
            pos_full=np.concatenate([pos_full, pos_halo2], axis=0)
            
        if bg is True:
            if (N1 is not None):
                rhob=3*N1/(4*mul_bg*np.pi*R1**3)*self.Om
                Nb=int(self.l**3*rhob)
                posb=np.random.random([Nb, 3])*self.l
                pos_full=np.concatenate([pos_full, posb], axis=0)
            elif (N2 is not None):
                rhob=3*N2/(4*mul_bg*np.pi*R2**3)*self.Om
                Nb=int(self.l**3*rhob)
                posb=np.random.random([Nb, 3])*self.l
                pos_full=np.concatenate([pos_full, posb], axis=0)
            elif Nb is not None:
                posb=np.random.random([Nb, 3])*self.l
                pos_full=np.concatenate([pos_full, posb], axis=0)
            else:
                print("provide a value of Nb or set bg to False!!")
                return 
        if(self.pbc is True):
            pos_full=pos_full%self.l
            
        return pos_full
    
##################################################################################################################
# this is not modified. No deletion applied here
    def create_filament_approx(self,t,xs, ys, zs, Nf,s, start_spine=None,
                    N1=None, c1=None, R1=None, N2=None, c2=None, R2=None, bg=False, mul_bg=200,Nb=None,seed=None):

        """Creates a realisation of a curved filament, with cylindrical radial profile, and NFW halos at both the ends,
        when the analytical function defining the spine is unkonwn, and the spine is defined by discretely sampled points.

        Inputs:
        t: array of the parameter describing the curve
        xs, ys, zs: arrays for coordinates of the spine corresponding to the t values
        The function linearly interpolates between these values to form a continuous spine.
        It finds the tangent using np.gradient(edge=2) method, and interpolates the tangent linearly as well.


        Nf: number of particles in the filament
        s: standard deviation of the gaussian describing azimuthal profile of the filament
        start_spine: staring point of the spine  (default is [l/2, l/2, l/2])

        N1, N2: number of particles in the nodes
        c1, c2: concentrations of the nodes (R200c/rs)
        R1, R2: R200c of the nodes
        bg: uniform background density if bg=True, otherwise background vacuum
        mul_bg: overdensity of the halos wrt critical density (default is 200)
        Nb: number of particles to be distributed uniformly in the background. 
        This must be given if there are no nodes, and bg=True
        
        Output: 
        cartesian coordinates of the positions of all the particles in the box 
             is a 2D array of shape [N,3]
             
        NOTE: if bg is True, the background particles are added on top of the gaussian filament, so that the 
        radial profile of the filament will be gaussian with a positive offset coming from the uniform background."""

        if seed is not None:
            np.random.seed(seed)
        if start_spine is None:
            start_spine=np.array([self.l/2, self.l/2, self.l/2])

        #creating the tangent vector
        tangx=np.gradient(xs, t, edge_order=2)
        tangy=np.gradient(ys, t, edge_order=2)
        tangz=np.gradient(zs, t, edge_order=2)

        tang_arr=np.array([tangx, tangy, tangz])
        pos_arr=np.array([xs, ys, zs])
        spine=pos_arr.copy()

        # length along the spine
        d=np.zeros_like(t)
        d0=0
        for i in range(1,len(d),1):
            d1=np.sqrt((xs[i]-xs[i-1])**2+(ys[i]-ys[i-1])**2+(zs[i]-zs[i-1])**2)
            d0=d0+d1
            d[i]=d0
        # writing t as a function of distance along the spine
        t_inter=interp1d(d, t)
        lcyl=np.max(d) # length of the filament

        #interpolting to write positions and tangent vectors as functions of t
        pos_spine=interp1d(t, pos_arr)
        tang_spine=interp1d(t, tang_arr)

        #################################################################################
        # creating a straight filament
        pos_cyl=self.create_cylinder(lcyl, s, Nf)

        # curving the filament
        # translate, rotate, translate
        B=np.array([0,0,1]) # originally, the filament is along z
        pos_new=pos_cyl-np.array([self.l/2, self.l/2, 0]) # taking the filament to x=y=0
        pos_curved=np.zeros_like(pos_cyl)
        for i in range(len(pos_curved)):
            #translate
            pos1=pos_new[i]-np.array([0,0,pos_cyl[i,2]]) # so that z value is 0 always
            d1=pos_new[i,2] #distance along the spine
            t1=t_inter(d1) #corresponding parameter value
            pos_c=pos_spine(t1) #position on the curved profile spine
            tang_c=tang_spine(t1)
            #rotate
            pos2=self.rotate(pos1, B, tang_c)
            #translate
            pos2=pos2+pos_c
            pos_curved[i]=pos2
        spine=spine.T
        pos_curved=pos_curved+(start_spine-spine[0]) # positioning the filament at the starting point   
        pos_full=pos_curved.copy()

        # translating the spine
        spine=spine+start_spine-spine[0]

        ####################################################################################
        # creating the NFW halos at the ends
        if ((N1 is not None)&(N2 is not None)):
            print("initial R2:", R2)
            R2=(N2/N1)**(1/3)*R1
            print("fixed R2:", R2)


        if ((N1 is not None)&(c1 is not None)&(R1 is not None)):
            print("creating first node")
            pos_halo1=self.create_NFW(c1, N=N1,R=R1, r0=spine[0])["pos_halo"]
            pos_full=np.concatenate([pos_full, pos_halo1], axis=0)

        if ((N2 is not None)&(c2 is not None)&(R2 is not None)):
            print("creating second node")
            pos_halo2=self.create_NFW(c2, N=N2,R=R2, r0=spine[-1])["pos_halo"]
            pos_full=np.concatenate([pos_full, pos_halo2], axis=0)
        if bg is True:
            if (N1 is not None):
                rhob=3*N1/(4*mul_bg*np.pi*R1**3)*self.Om
                Nb=int(self.l**3*rhob)
                posb=np.random.random([Nb, 3])*self.l
                pos_full=np.concatenate([pos_full, posb], axis=0)
            elif (N2 is not None):
                rhob=3*N2/(4*mul_bg*np.pi*R2**3)*self.Om
                Nb=int(self.l**3*rhob)
                posb=np.random.random([Nb, 3])*self.l
                pos_full=np.concatenate([pos_full, posb], axis=0)
            elif Nb is not None:
                posb=np.random.random([Nb, 3])*self.l
                pos_full=np.concatenate([pos_full, posb], axis=0)
            else:
                print("provide a value of Nb or set bg to False!!")
                return 
            
        if (self.pbc is True):
            pos_full=pos_full%self.l


        return pos_full
    
    ##############################################################################################
     # background subtracted gaussian distribution, so that after adding the background, the filament has gaussian radial profile
    def r_trunc(self,s, cf):
        '''returns the value of r at which the gaussian density equals the background density
        where
        s: sigma of the gaussian
        cf: filament concentration'''
        
        Rf=np.sqrt(2*s**2*np.log(cf))
        return Rf
    
    def gauss_rho_bg(self,r,s, cf):
        '''background subtracted gaussian profile, so that adding uniform background creates a gaussian profile
        cf is the concentration. i.e, central density/bg density
        Note that the profile is not nromalised
        Also, this expression is valid only for r<r_trunc'''
        y=cf*np.exp(-r**2/(2*s**2))-1
        return y

    #normalised pdf (\propto r*rho(r) for cylindrical profile)
    def gauss_pdf_bg(self,r,s, cf):
        '''background subtracted gaussian pdf, not normalised'''
        y= r*(cf*np.exp(-r**2/(2*s**2))-1)
        return y

    def gauss_cdf_bg(self,r,s, cf):
        '''normalised background subtracted cdf. Should be evaluated for r<r_trunc'''
        Rf=self.r_trunc(s, cf)
        
        y=cf*s**2*(1-np.exp(-r**2/(2*s**2)))-r**2/2
        y1=y/(cf*s**2*(1-np.exp(-Rf**2/(2*s**2)))-Rf**2/2)
        return y1
    
    def create_cylinder_bg(self,lcyl, s,cf, N,Nsamp=int(1e5), seed=None):
        """ creates a realisation of background subtracted cylindrical gaussian profile, 
        truncated at Rf, where the filament density equals background density.
        
        Inputs:
        lcyl: length of the cylinder
        s: sigma of the radial density profile
        cf: filament concentration
        N: number of particles in the filament
        Nsamp: number of sampling points used while interpolating the Gaussian CDF. (Default is int(1e5))
        seed: random seed
        
        Output:
        The array of [x, y,z] positions of N particle in the shape[N, 3]
        The cylinder spine is along the z axis, with x=y=lbox/2, z_init=0
        
        NOTE: This cannot be used to generate velocities, use create_cylinder_bg1 instead
        But, if velocities are not requried, this function is faster."""
        
        if seed is not None:
            np.random.seed(seed)
        
        Rf=self.r_trunc(s, cf)
        r0=np.logspace(-5,np.log10(Rf), Nsamp) # values of r to interpolate cdf
        
        CDF=self.gauss_cdf_bg(r0, s, cf) # array of cdf values for given sample of r0 values
        r_interp=interp1d(CDF, r0) # function providing r for given cdf value
        
        if (s<self.l/5):
            cdf1=np.random.random(N) # uniform random sampling of the CDF
            r=r_interp(cdf1)

            phi=np.random.random(N)*np.pi*2 # uniform random phi
            z=np.random.random(N)*lcyl # uniform random z

            x=r*np.cos(phi)
            y=r*np.sin(phi)

        else:
            # First generate 100*N as many points as required,
            #retain only those which lie in the box, then keep only N such points

            cdf1=np.random.random(100*N) # uniform random sampling of the CDF
            r=r_interp(cdf1)

            phi=np.random.random(100*N)*np.pi*2 # uniform random phi
            z=np.random.random(100*N)*lcyl # uniform random z

            x=r*np.cos(phi)
            y=r*np.sin(phi)

            #assuming the filament to be at the center of the box
            ind=np.where((np.abs(x)<=self.l/2)&(np.abs(y)<=self.l/2))[0]
            if (len(ind)<N):
                print("not enough points, retry")
            else:   
                x=x[ind]
                y=y[ind]
                z=z[ind]

                #retaining only N random points
                x=x[:N]
                y=y[:N]
                z=z[:N]
        x=x+self.l/2
        y=y+self.l/2
        pos_cyl=np.array([x, y, z]).T
        return pos_cyl
    
    ######################################################################################################################
    def create_filament_bg(self,pos_spine, tang_spine, rho0,s,cf, start_spine=None,
                           c1=None, R1=None,c2=None, R2=None, bmul_bg=200, Nsamp=int(1e5),seed=None):
        """Creates a realisation of a curved filament, with Gaussian radial profile, and uniform background.
        The Gaussian is truncated at Rf, where the filament density reaches background density.
        Inputs:
        pos_spine: function descrbing the position of the spine as a parametric curve
        tang_spine: derivative of pos_spine
        these functions are functions of the type f(t), which returns [x(t), y(t), z(t)] and [x'(t), y'(t), z'(t)] respectively.
        where t runs from [0,1] 
        
        rho0: background number density
        s: standard deviation of the gaussian describing azimuthal profile of the filament
        cf: concentration of the filament (density at center/bg density)
        start_spine: staring point of the spine  (default is [l/2, l/2, l/2])
        
        c1, c2: concentrations of the nodes (R200c/rs)
        R1, R2: R200c of the nodes
        mul_bg: overdensity of the halo wrt critical density (Default is 200)
        Nsamp: number of sampling points while interpolating the gaussian cdf (Default int(1e5))
        seed: random seed
        
        output:
        pos_full: 2D array in the shape [N, 3], giving the cartesian position coordinates of the the particles"""
        
        if seed is not None:
            np.random.seed(seed)
        if start_spine is None:
            start_spine=np.array([self.l/2, self.l/2, self.l/2])
         
        # length along the spine
        dt=0.0001
        t=np.arange(0, 1+dt, dt)
        spine=pos_spine(t)
        
        d=np.zeros_like(t)
        d0=0
        for i in range(1,len(d),1):
            d1=np.sqrt((spine[0][i]-spine[0][i-1])**2+(spine[1][i]-spine[1][i-1])**2+(spine[2][i]-spine[2][i-1])**2)
            d0=d0+d1
            d[i]=d0
        # writing t as a function of distance along the spine
        t_inter=interp1d(d, t)
        lcyl=np.max(d) # length of the filament
        ###############################################################################
        # calculating the number of particles in each component
        Rf=self.r_trunc(s, cf)
        Nf=int(rho0*(cf*lcyl*2*np.pi*s**2*(1-np.exp(-Rf**2/(2*s**2)))-np.pi*Rf**2*lcyl))
        Nb=int(rho0*self.l**3) 
        N1=0 # number of particles in the NFW halo 1
        N2=0
        N1sub=0
        N2sub=0
        if (R1 is not None):
            N1=int(rho0*(4/3)*np.pi*R1**3*bmul_bg/self.Om)
            N1sub=int(rho0*(4/3)*np.pi*R1**3)
        if (R2 is not None):
            N2=int(rho0*(4/3)*np.pi*R2**3*bmul_bg/self.Om)
            N2sub=int(rho0*(4/3)*np.pi*R2**3)
        print("total number of particles:", Nf+Nb-N1sub-N2sub+N1+N2)
        print("Nf:", Nf, "\nNb:", Nb-N1sub-N2sub, "\nN1:", N1, "\nN2:", N2) 
        
        #################################################################################
        # creating a straight filament
        pos_cyl=self.create_cylinder_bg(lcyl, s,cf, Nf, Nsamp, seed)
        
        # curving the filament
        # translate, rotate, translate
        B=np.array([0,0,1]) # originally, the filament is along z
        pos_new=pos_cyl-np.array([self.l/2, self.l/2, 0]) # taking the filament to x=y=0
        pos_curved=np.zeros_like(pos_cyl)
        for i in range(len(pos_curved)):
            #translate
            pos1=pos_new[i]-np.array([0,0,pos_cyl[i,2]]) # so that z value is 0 always
            d1=pos_new[i,2] #distance along the spine
            t1=t_inter(d1) #corresponding parameter value
            pos_c=pos_spine(t1) #position on the curved profile spine
            tang_c=tang_spine(t1)
            #rotate
            pos2=self.rotate(pos1, B, tang_c)
            #translate
            pos2=pos2+pos_c
            pos_curved[i]=pos2
        spine=spine.T
        pos_curved=pos_curved+(start_spine-spine[0]) # positioning the filament at the starting point   
        pos_full=pos_curved.copy()
        
        # translating the spine
        spine=spine+start_spine-spine[0]
        
        ####################################################################################
        # creating the NFW halos at the ends        
        if ((c1 is not None)&(R1 is not None)):
            print("creating first node")
            pos_halo1=self.create_NFW(c1, N=N1,R=R1, r0=spine[0], seed=seed+1)["pos_halo"]
            
            # deleting filament particles inside the halo
            ind=np.where(np.linalg.norm(pos_full-spine[0], axis=1)<R1)[0]
            pos_full=np.delete(pos_full, ind, axis=0)
            pos_full=np.concatenate([pos_full, pos_halo1], axis=0)
            
        if ((c2 is not None)&(R2 is not None)):
            print("creating second node")
            pos_halo2=self.create_NFW(c2, N=N2,R=R2, r0=spine[-1], seed=seed+2)["pos_halo"]   
            # deleting filament particles inside the halo
            ind=np.where(np.linalg.norm(pos_full-spine[-1], axis=1)<R2)[0]
            pos_full=np.delete(pos_full, ind, axis=0)
            pos_full=np.concatenate([pos_full, pos_halo2], axis=0)
            
        #creating uniform background    
        posb=np.random.random([Nb, 3])*self.l
        
        # removing background particles on top of halos
        if (N1!=0):
            ind=np.where(np.linalg.norm(posb-spine[0], axis=1)<R1)[0]
            posb=np.delete(posb, ind, axis=0)
        if (N2!=0):
            ind=np.where(np.linalg.norm(posb-spine[-1], axis=1)<R2)[0]
            posb=np.delete(posb, ind, axis=0)
        
        
        pos_full=np.concatenate([pos_full, posb], axis=0)
        
        if (self.pbc is True):
            pos_full=pos_full%self.l
            
        return pos_full
    ###################################################################################################################
    # another way of creating uniform backgroud, and generating velocity profiles as well
    # this is done by deleting background particles on top of the filament
    def gauss_rho_bg1(self,r,s, cf):
        '''gaussian profile
        cf is the concentration. i.e, central density/bg density
        Note that the profile is not nromalised
        Also, this expression is valid only for r<r_trunc'''
        y=cf*np.exp(-r**2/(2*s**2))
        return y

    #normalised pdf (\propto r*rho(r) for cylindrical profile)
    def gauss_pdf_bg1(self,r,s, cf):
        '''gaussian pdf, not normalised'''
        y= r*(cf*np.exp(-r**2/(2*s**2)))
        return y

    def gauss_cdf_bg1(self,r,s, cf):
        '''normalised gaussian cdf. Should be evaluated for r<r_trunc'''
        Rf=self.r_trunc(s, cf)
        
        y=cf*s**2*(1-np.exp(-r**2/(2*s**2)))
        y1=y/(cf*s**2*(1-np.exp(-Rf**2/(2*s**2))))
        return y1
    
    
    def create_cylinder_bg1(self,lcyl, s,cf, N, Nsamp=int(1e5),seed=None, 
                           vel=False,sigvb=None, vz_mean_prof=None, vr_mean_prof=None, 
                            vz_std_prof=None, vr_std_prof=None, vphi_std_prof=None):
        """ generates a realisation of a cylindrical gaussian profile, (position as well as velocity data),
        for the specified velocity and density models. 
        The cylinder spine is along the z axis, with x=y=lbox/2, z_init=0
        Inputs:
        lcyl: cylinder length
        s: gaussian sigma
        cf: filament central concentration
        N: number of particles
        Nsamp: number of sampling points for interpolating Gaussian CDF (Default is int(1e5))
        seed: random number generation seed
        vel: True if velocity information is to be generated (Default is False)
        
        if vel is True, following information is needed:
        sigvb: bakcground 1d velocity dispersion
        Following functions of the form f(r, z, phi)
        vz_mean_prof: mean vz at given (r,z,phi)
        vr_mean_prof: mean vr at give (r,z, phi)
        vz_std_prof: z-velocity disperison at given (r,z, phi)
        vr_std_prof: radial velocity dispersion at given (r, z, phi)
        vphi_std_prof: azimuthal velocity dispersion at given (r,z, phi)
        
        Outputs:
        A dictionary D, with following keys:
        'pos_cyl': cartesian position coordinates, 2D array of shape [N,3]
        'vel_cyl': cartesian velocity coordiantes, 2D array of shape [N,3]
        """
        
        if seed is not None:
            np.random.seed(seed)

        Rf=self.r_trunc(s, cf)
        cent=lcyl/2 #center of the filament
        r0=np.logspace(-5,np.log10(Rf), Nsamp) # values of r to interpolate cdf
        
        CDF=self.gauss_cdf_bg1(r0, s, cf) # array of cdf values for given sample of r0 values
        r_interp=interp1d(CDF, r0) # function providing r for given cdf value
        
        if (s<self.l/5):
            cdf1=np.random.random(N) # uniform random sampling of the CDF
            r=r_interp(cdf1)

            phi=np.random.random(N)*np.pi*2 # uniform random phi
            z=np.random.random(N)*lcyl # uniform random z

            x=r*np.cos(phi)
            y=r*np.sin(phi)

        else:
            # First generate 100*N as many points as required,
            #retain only those which lie in the box, then keep only N such points

            cdf1=np.random.random(100*N) # uniform random sampling of the CDF
            r=r_interp(cdf1)

            phi=np.random.random(100*N)*np.pi*2 # uniform random phi
            z=np.random.random(100*N)*lcyl # uniform random z

            x=r*np.cos(phi)
            y=r*np.sin(phi)

            #assuming the filament to be at the center of the box
            ind=np.where((np.abs(x)<=self.l/2)&(np.abs(y)<=self.l/2))[0]
            if (len(ind)<N):
                print("not enough points, retry")
            else:   
                x=x[ind]
                y=y[ind]
                z=z[ind]

                #retaining only N random points
                x=x[:N]
                y=y[:N]
                z=z[:N]
                
                r=np.sqrt(x**2+y**2)
                phi=phi[:N]
        

        if (vel is True):
            print("vz at center:", vz_mean_prof(0,cent,0))
            v_r=np.random.normal(loc=vr_mean_prof(r, z, phi), scale=vr_std_prof(r, z, phi))
            v_z=np.random.normal(loc=vz_mean_prof(r, z, phi), scale=vz_std_prof(r, z, phi))
            v_phi=np.random.normal(loc=0.0, scale=vphi_std_prof(r, z, phi))
            
            v_x=v_r*np.cos(phi)-v_phi*np.sin(phi)
            v_y=v_r*np.sin(phi)+v_phi*np.cos(phi)
            
            vel_cyl=np.array([v_x, v_y, v_z]).T
        else:
            vel_cyl=0
            
        x=x+self.l/2
        y=y+self.l/2
        pos_cyl=np.array([x, y, z]).T
        
        D=dict()
        D["pos_cyl"]=pos_cyl
        D["vel_cyl"]=vel_cyl
        
        return D
    
    ######################################################################################################################
    def create_filament_bg1(self,pos_spine, tang_spine, rho0,s,cf, start_spine=None,
                            c1=None, R1=None,c2=None, R2=None, bmul_bg=200, Nsamp=int(1e5),seed=None,
                             vel=False,sigvb=None, vz_mean_prof=None, vr_mean_prof=None, 
                            vz_std_prof=None, vr_std_prof=None, vphi_std_prof=None, beta=0,
                            
                            cdf_r_2h1=None, mul_2h1=10,
                            vr_mean_prof_2h1=None, vr_std_prof_2h1=None,
                            vphi_mean_prof_2h1=None, vphi_std_prof_2h1=None,
                            vtheta_mean_prof_2h1=None, vtheta_std_prof_2h1=None,
                           
                           cdf_r_2h2=None, mul_2h2=10,
                            vr_mean_prof_2h2=None, vr_std_prof_2h2=None,
                            vphi_mean_prof_2h2=None, vphi_std_prof_2h2=None,
                            vtheta_mean_prof_2h2=None, vtheta_std_prof_2h2=None):
        """Generates a realisation of the given model of filament (both positions and velocities),
        where the radial density profile is a Gaussian.
        The NFW halos are truncated at R200c
        The halo outskirts are from R200c to 4*R200b
        
        Inputs:
        pos_spine: function descrbing the position of the spine as a parametric curve
        tang_spine: derivative of pos_spine
        these functions are functions of the type f(t), which returns [x(t), y(t), z(t)] and [x'(t), y'(t), z'(t)] respectively.
        where t runs from [0,1] 
        
        rho0: background number density
        s: standard deviation of the gaussian describing azimuthal profile of the filament
        cf: concentration of the filament (density at center/bg density)
        start_spine: staring point of the spine  (default is [l/2, l/2, l/2])
        
        c1, c2: concentrations of the nodes (R200c/rs)
        R1, R2: R200c of the nodes
        bmul_bg: overdensity of the halo wrt critical density (default is 200)
        Nsamp: number of sampling points for interpolating Gaussian CDF (Default is int(1e5))
        
        vel: True if velocity information is to be generated (Default is False)
        if vel is True, following information is needed:
        sigvb: bakcground 1d velocity dispersion
        Following functions of the form f(r, z, phi)
        vz_mean_prof: mean vz at given (r,z,phi)
        vr_mean_prof: mean vr at give (r,z, phi)
        vz_std_prof: z-velocity disperison at given (r,z, phi)
        vr_std_prof: radial velocity dispersion at given (r, z, phi)
        vphi_std_prof: azimuthal velocity dispersion at given (r,z, phi)
        beta: velocity anisotropy of NFW halo(default 0)
        
        Halo outskirts profiles:
        Here, r is the radius in spherical polar coordinates.
        cdf_r_2h: cdf(r) obtained from the radial density profile in the halo outskirst. 
        if set to None, this region is treated as background.
        mul_2h= enclosed desnsity in the halo outskirts as compared to critical density (set to 10)
        everything with prof_2h: velocity profiles in the halo outskirst, of the functional form f(r, theta, phi)
        
        Outputs:
        Dictionary D with the following keys:
        'pos_full': cartesian position coordinates, 2D array of shape [N,3]
        'vel_full': cartesian velocity coordiantes, 2D array of shape [N,3]
        'Rf:' edge of the filament
        """
        
        if seed is not None:
            np.random.seed(seed)
        if start_spine is None:
            start_spine=np.array([self.l/2, self.l/2, self.l/2])
         
        # length along the spine
        dt=1e-5
        t=np.arange(0, 1+dt, dt)
        spine=pos_spine(t)
        
        d=np.zeros_like(t)
        d0=0
        for i in range(1,len(d),1):
            d1=np.sqrt((spine[0][i]-spine[0][i-1])**2+(spine[1][i]-spine[1][i-1])**2+(spine[2][i]-spine[2][i-1])**2)
            d0=d0+d1
            d[i]=d0
        # writing t as a function of distance along the spine
        t_inter=interp1d(d, t)
        lcyl=np.max(d) # length of the filament
        cent=lcyl/2
        ###############################################################################
        # calculating the number of particles in each component
        Rf=self.r_trunc(s, cf)
        Nf=int(rho0*cf*lcyl*2*np.pi*s**2*(1-np.exp(-Rf**2/(2*s**2))))
        Nb=int(rho0*self.l**3) 
        N1=0 # number of particles in the NFW halo 1
        N2=0
        N1sub=0
        N2sub=0
        
        N1out=0
        N2out=0
        
        def sol(w, c):
            y1=np.log(1+w*c)- w*c/(w*c+1)
            y2=np.log(1+c)-c/(1+c)
            y3=w**3*self.Om

            y4=y1/y2-y3
            return y4                
            
        if (R1 is not None):
            N1=int(rho0*(4/3)*np.pi*R1**3*bmul_bg/self.Om)
            N1sub=int(rho0*(4/3)*np.pi*R1**3)
            
            if (cdf_r_2h1 is not None):
                w1=fsolve(sol, 1.1, args=(c1))
                R1out=4*w1*R1
                
                N1out=int(rho0*(4/3)*np.pi*(R1out**3-R1**3)*mul_2h1/self.Om)
                N1sub=int(rho0*(4/3)*np.pi*R1out**3)
                          
        if (R2 is not None):
            N2=int(rho0*(4/3)*np.pi*R2**3*bmul_bg/self.Om)
            N2sub=int(rho0*(4/3)*np.pi*R2**3)
            
            if (cdf_r_2h2 is not None):
                w2=fsolve(sol, 1.1, args=(c2))
                R2out=4*w2*R2

                N2out=int(rho0*(4/3)*np.pi*(R2out**3-R2**3)*mul_2h2/self.Om)
                N2sub=int(rho0*(4/3)*np.pi*R2out**3)
                          
                
        print("total number of particles:", Nf+Nb-N1sub-N2sub+N1+N2+N1out+N2out)
        
        #################################################################################
        # creating a straight filament
        D_cyl=self.create_cylinder_bg1(lcyl, s,cf, Nf, Nsamp=Nsamp,seed=seed,
                                       vel=vel,sigvb=sigvb,
                                       vz_mean_prof=vz_mean_prof, vr_mean_prof=vr_mean_prof,
                                      vz_std_prof=vz_std_prof, vr_std_prof=vr_std_prof, vphi_std_prof=vphi_std_prof)

        pos_cyl=D_cyl["pos_cyl"]
        vel_cyl=D_cyl["vel_cyl"]
        
        # curving the filament
        # translate, rotate, translate
        B=np.array([0,0,1]) # originally, the filament is along z
        pos_new=pos_cyl-np.array([self.l/2, self.l/2, 0]) # taking the filament to x=y=0
        pos_curved=np.zeros_like(pos_cyl)
        if (vel is True):
            vel_curved=np.zeros_like(vel_cyl)
            
        for i in range(len(pos_curved)):
            #translate
            pos1=pos_new[i]-np.array([0,0,pos_cyl[i,2]]) # so that z value is 0 always
            d1=pos_new[i,2] #distance along the spine
            t1=t_inter(d1) #corresponding parameter value
            pos_c=pos_spine(t1) #position on the curved profile spine
            tang_c=tang_spine(t1)
            #rotate
            pos2=self.rotate(pos1, B, tang_c)
            if (vel is True):
                vel1=vel_cyl[i]
                vel2=self.rotate(vel1, B, tang_c)
                vel_curved[i]=vel2
            #translate
            pos2=pos2+pos_c
            pos_curved[i]=pos2
        spine=spine.T
        pos_curved=pos_curved+(start_spine-spine[0]) # positioning the filament at the starting point   
        pos_full=pos_curved.copy()
        if (vel is True):
            vel_full=vel_curved.copy()
        
        # translating the spine
        spine=spine+start_spine-spine[0]
        
        ####################################################################################
        # creating the NFW halos at the ends        
        if ((c1 is not None)&(R1 is not None)):
            print("creating first node")
            D1=self.create_NFW(c1, N=N1,R=R1, r0=spine[0],vel=vel, beta=beta, seed=seed+1)
            pos_halo1=D1["pos_halo"]
            # deleting filament inside the halo
            ind=np.where(np.linalg.norm(pos_full-spine[0], axis=1)<R1)[0]
            pos_full=np.delete(pos_full, ind, axis=0)
            print("\n number of particles of filament deleted inside halo1:", len(ind))
            pos_full=np.concatenate([pos_full, pos_halo1], axis=0)
            if (vel is True):
                vel_halo1=D1["vel_halo"]
                vel_full=np.delete(vel_full, ind, axis=0)
                vel_full=np.concatenate([vel_full, vel_halo1], axis=0)
            
            
        if ((c2 is not None)&(R2 is not None)):
            print("creating second node")
            D2=self.create_NFW(c2, N=N2,R=R2, r0=spine[-1], vel=vel, beta=beta, seed=seed+2)
            pos_halo2=D2["pos_halo"]
            ind=np.where(np.linalg.norm(pos_full-spine[-1], axis=1)<R2)[0]
            pos_full=np.delete(pos_full, ind, axis=0)
            print("\n number of particles of filament deleted inside halo2:", len(ind))
            pos_full=np.concatenate([pos_full, pos_halo2], axis=0)
            if (vel is True):
                vel_halo2=D2["vel_halo"]
                vel_full=np.delete(vel_full, ind, axis=0)
                vel_full=np.concatenate([vel_full, vel_halo2], axis=0)
         
        
        #creating halo outskirts
        if ((c1 is not None)&(R1 is not None)&(cdf_r_2h1 is not None)):
            print("creating first halo outskirts")
            D1=self.create_halo_outskirts(c=c1, cdf_r_2h=cdf_r_2h1, 
                              vr_mean_prof_2h=vr_mean_prof_2h1, vr_std_prof_2h=vr_std_prof_2h1,
                            vphi_mean_prof_2h=vphi_mean_prof_2h1, vphi_std_prof_2h=vphi_std_prof_2h1,
                            vtheta_mean_prof_2h=vtheta_mean_prof_2h1, vtheta_std_prof_2h=vtheta_std_prof_2h1, 
                              N=N1out,R=R1, r0=spine[0], seed=seed+3, vel=vel)

            pos_2h1=D1["pos_2h"]
            print("4R200m for the first halo:", D1["R2h"])
            pos_full=np.concatenate([pos_full, pos_2h1], axis=0)
            if (vel is True):
                vel_2h1=D1["vel_2h"]
                vel_full=np.concatenate([vel_full, vel_2h1], axis=0)
                
        if ((c2 is not None)&(R2 is not None)&(cdf_r_2h2 is not None)):
            print("creating second halo outskirts")
            D1=self.create_halo_outskirts(c=c2, cdf_r_2h=cdf_r_2h2, 
                              vr_mean_prof_2h=vr_mean_prof_2h2, vr_std_prof_2h=vr_std_prof_2h2,
                            vphi_mean_prof_2h=vphi_mean_prof_2h2, vphi_std_prof_2h=vphi_std_prof_2h2,
                            vtheta_mean_prof_2h=vtheta_mean_prof_2h2, vtheta_std_prof_2h=vtheta_std_prof_2h2, 
                              N=N2out,R=R2, r0=spine[-1], seed=seed+4, vel=vel)
            pos_2h2=D1["pos_2h"]
            print("4R200m for the second halo:", D1["R2h"])
            pos_full=np.concatenate([pos_full, pos_2h2], axis=0)
            if (vel is True):
                vel_2h2=D1["vel_2h"]
                vel_full=np.concatenate([vel_full, vel_2h2], axis=0)
            

        #creating uniform background
        posb=np.random.random([Nb, 3])*self.l
        
        # removing background particles on top of halos and outskirts
        if (N1out!=0):
            ind=np.where(np.linalg.norm(posb-spine[0], axis=1)<R1out)[0]
            posb=np.delete(posb, ind, axis=0)
            print("number of particles deleted on top of halo1 and outskirts:",len(ind))
        elif (N1!=0):
            ind=np.where(np.linalg.norm(posb-spine[0], axis=1)<R1)[0]
            posb=np.delete(posb, ind, axis=0)
            print("number of particles deleted on top of halo1:",len(ind))
        if (N2out!=0):
            ind=np.where(np.linalg.norm(posb-spine[-1], axis=1)<R2out)[0]
            posb=np.delete(posb, ind, axis=0)
        elif (N2!=0):
            ind=np.where(np.linalg.norm(posb-spine[-1], axis=1)<R2)[0]
            posb=np.delete(posb, ind, axis=0)
            
        # removing background particles on top of the filament
        # choosing particles in a cuboid around the spine (those outside are never on top of the filament)
        boxmax=np.max(spine, axis=0)+Rf
        boxmin=np.min(spine, axis=0)-Rf
        
        IND=np.arange(len(posb)) # indices of the original background particles
        ind1=np.where((posb[:,0]>=boxmin[0])&(posb[:,0]<=boxmax[0]))[0]
        posb1=posb[ind1]
        
        posb_r1=np.delete(posb, ind1, axis=0) # remaining particles in the backgroud, not to be deleted
        
        ind2=np.where((posb1[:,1]>=boxmin[1])&(posb1[:,1]<=boxmax[1])&
                      (posb1[:,2]>=boxmin[2])&(posb1[:,2]<=boxmax[2]))[0]
        posb2=posb1[ind2] # particles to be checked
        posb_r2=np.delete(posb1, ind2, axis=0) # remaining particles in the background

        
        ind_del=[]
        for p in range(len(posb2)):
            p_this=posb2[p]
            d=np.sqrt((p_this[0]-spine[:,0])**2+(p_this[1]-spine[:,1])**2+(p_this[2]-spine[:,2])**2)
            
            ind_min=np.min(np.where(d==np.min(d))[0])
            if ((np.min(d)<=Rf)&(ind_min!=0)&(ind_min!=(len(d)-1))):
                # the last 2 conditions so that hemispheres are not deleted at the ends of the filament
                ind_del.append(p)
        ind_del=np.array(ind_del, dtype=int)
        
        print("number of particles deleted on top of the filament:", len(ind_del))
#         print("expected number:", np.pi*Rf**2*lcyl/(self.l**3)*Nb)
        posb2=np.delete(posb2, ind_del, axis=0)
        
        posb=np.concatenate([posb_r1, posb_r2, posb2], axis=0) # filament subtracted background
        
        if (vel is True):
            velb=np.random.normal(scale=sigvb,size=[len(posb),3])
            vel_full=np.concatenate([vel_full, velb], axis=0)
        else:
            vel_full=0

        pos_full=np.concatenate([pos_full, posb], axis=0)
        
        if (self.pbc is True):
            pos_full=pos_full%self.l
        D=dict()
        D["vel_full"]=vel_full
        D["pos_full"]=pos_full
        
        D["cent"]=cent
        D["Rf"]=Rf
            
        return D
    