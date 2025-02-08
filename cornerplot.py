import numpy as np
import matplotlib.pyplot as plt
import sys

# note that np.cov() has N-1 in the denominator (unbiased estimate)
class Fisher(object):
    """ Calculates covariance matrices, model derivatives, fisher information matrix, and plots the corner plots"""
    def __init__(self, Nreal=100, m=13, N=6):
        """ Nreal: number of realisations
        m: dimensions of the data vector
        N: number of model parameters"""
        self.Nreal=Nreal
        self.m=m
        self.N=N

            
    def calc_cov(self, data, return_norm=False, Hartlap_correct=False, jk=False):
        """ Given the m dimensional data from Nreal realisations, the function calculates the covariance matrix.
        inputs:
        data: [Nreal, m] dimensional matrix
        return_norm: if True, returns the normalised covariance matrix as well. (default is False)
        Hartlap_correct: if True, inverse covariance is mulitplied by (Nreal-m-2)/(Nreal-1), according to Hartlap+2006. (default is False)
        jk: True if the input data is jack knife averages. Then the Covariance matrix is multiplied by (Nreal-1)^2/Nreal. (default is False)
        output:
        Dictionary D with following keys:
        C, CI, mu; where:
        C: Covariance matrix (with Nreal-1 factor in the denominator)
        CI: inverse covariance
        mu: mean
        
        if return_norm is True
        NC: normalised covariance matrix (C_ij/sigma_i sigma_j)"""
        Nreal=len(data)
        m=len(data[0])

        if (self.m!=m):
            print("The data[0] dimensions do not match the input number of data vector dimensions m. Proceeding by reassigning the dimension based on length of data[0].")
            self.m=m

        if (self.Nreal!=Nreal):
            print("The data dimensions do not match the input number of realisations Nreal. Proceeding by reassigning the number of realisations based on length of the data.")
            self.Nreal=Nreal

        C=np.cov(data.T)

        if (jk is True):
            C*=(Nreal-1)**2/Nreal

        mu=np.mean(data, axis=0)
        CI=np.linalg.inv(C)

        if (Hartlap_correct is True):
            CI=CI*(self.Nreal-self.m-2)/(self.Nreal-1)

    
        D={}
        D["C"]=C
        D["CI"]=CI
        D["mu"]=mu
    
        if (return_norm is True):
            NC=np.zeros_like(C)
            for i in range(len(C)):
                for j in range(i, len(C), 1):
                    NC[i][j]=NC[j][i]=C[i][j]/np.sqrt(C[i][i]*C[j][j])
            D["NC"]=NC
    
        return D
    
    def calc_deriv(self,data, d_theta):
        
        """ calculates the model derivatives using finite difference method, using 3-point or 5-point method.
        inputs:
        data: array of size [m, d], where the model output is m-dimensional, and d is number of points (3 or 5). 
        dtheta: If theta is the parameter being varied, the data is at [theta0 - 2d_theta, theta0 - d_theta, theta0, theta0 + d_theta, theta0 + 2d_theta] for d=5, central 3 for d=3.
        
        output: dmdtheta array of size m """
        m=len(data)
        d=len(data[0])

        if (self.m!=m):
            print("The length of data does not match initiated data vector dimensions. Proceeding by reassigning the length of data to be the new data vector dimensions")
            self.m=m

        if ((d!=3)&(d!=5)):
            print("this function can calculate only 3 or 5 point derivatives. %d points are provided!"%d)
            dmdtheta = -1
        elif (d==5):
            dmdtheta = (data[:,0]-8*data[:,1]+8*data[:,3]-data[:,4])/(12*d_theta)
        else:
            dmdtheta = (data[:,2]-data[:,0])/(2*d_theta)
        return dmdtheta
    
    def calc_fisher(self, CI, dmdtheta ):
        """calculates the fisher matrix and its inverse given the inverse covariance matrix and model derivatives.
        input:
        CI: inverse covariance matrix shape [ m, m]
        dmdtheta: model derivatives shape [N, m]
        where m: length of data vector, N: number of parameters
        output: Dictionary D with following keys:
        F: Fisher matrix of size [N, N]
        FI: Inverse fisher matrix of size [N, N]"""
    
        f1=np.matmul(CI, dmdtheta.T)
        F=np.matmul(dmdtheta, f1)
        FI=np.linalg.inv(F)
        D={}
        D["F"]=F
        D["FI"]=FI
        return D
    
    def plot_corner(self, C,mu, names=None, CL=[0.683, 0.954, 0.9973], ls1=["-", "--", ":"], 
            fillcol="mediumpurple",fillalph=0.2 , fig=None, ax=None, legend=True, label=None, sig_yval=0.8):
        """ Plots the corner plots given the inverse Fisher matrix C (or the covariance matrix of the parameters).
        Inputs:
        C: N x N parameter covariance matrix
        mu: fiducial parameter values (array of length N)
        names: labels of the parameters (list of length N)
        CL: list of confidence levels to be plotted (0<CL<1). Default is [0.683, 0.954, 0.9973])
        ls1: line styles for each CL. Ensure len(ls1)=len(CL)
        fillcol: colour to fill the contours (default is "purple")
        fillalph: alpha of the fill (default is 0.2)
        fig, ax: None by default, then create new figure. Else plot on the given figure.
        legend: True or False (default is True)
        label: Label for this set of corner plots, in case multiple corner plots are shown on the same figure. (default is None)
        sig_yval: value of y at which the width of the gaussians is reported.
        Outputs the figure and axes"""
        if label is None:
            label=" "
        if (len(C)!=len(mu)):
            print("lenght of C does not match length of mu, exiting")
            return -1
        if (len(CL)!=len(ls1)):
            print("length of CL does not match length of line styles ls1, exiting")
            return -1
        CL=np.array(CL)
        alph=np.sqrt(-2*np.log(1-CL)) # scaling factor of ellipse
    
        if ((fig is None)+(ax is None)):
            fig, ax=plt.subplots(len(C),len(C), figsize=(15,15))
        elif(len(ax)!=len(C)):
            fig, ax=plt.subplots(len(C),len(C), figsize=(15,15))
        for i in range(len(C)):
            sig=np.sqrt(C[i][i])
            x=np.linspace(-5*sig, 5*sig, 100)+mu[i]
            y=np.exp(-(x-mu[i])**2/(2*sig**2))/np.sqrt(2*np.pi*sig**2)
            ax[i][i].plot(x,y , c=fillcol, label=label)
            ax[i][i].vlines(mu[i],0, np.max(y), color="slateblue")
            ax[i][i].vlines(mu[i]+sig, 0, np.max(y), ls=":", color=fillcol)
            ax[i][i].vlines(mu[i]-sig, 0, np.max(y), ls=":", color=fillcol)
            ax[-1][i].set_xlabel(names[i])
            ax[i][0].set_ylabel(names[i])
            ax[i][i].set_yticklabels([])
            ax[i][i].set_xlim(mu[i]-5*sig, mu[i]+5*sig)
            ax[i][i].text(0.6, sig_yval,"%0.2e"%sig, transform=ax[i][i].transAxes, color=fillcol, fontsize=15)
    
    
            for j in range(i+1,len(C), 1):
                sigx=np.sqrt(C[i][i])
                sigy=np.sqrt(C[j][j])
                sigxy=C[i][j]
    
                u=mu[i]
                v=mu[j]
    
                a0=np.sqrt((sigx**2+sigy**2)/2+np.sqrt((sigx**2-sigy**2)**2/4+sigxy**2)) #semimajor axis
                b0=np.sqrt((sigx**2+sigy**2)/2-np.sqrt((sigx**2-sigy**2)**2/4+sigxy**2)) #semiminor axis
                t_rot=0.5*(np.arctan2(2*sigxy,(sigx**2-sigy**2))) # angle
    
                t = np.linspace(0, 2*np.pi, 100)
                for m in range(len(alph)):
                    a=alph[m]*a0
                    b=alph[m]*b0
    
                    Ell = np.array([a*np.cos(t) , b*np.sin(t)])
                    R_rot = np.array([[np.cos(t_rot) , -np.sin(t_rot)],[np.sin(t_rot) , np.cos(t_rot)]])
    
                    Ell_rot = np.zeros((2,Ell.shape[1]))
                    for k in range(Ell.shape[1]):
                        Ell_rot[:,k] = np.dot(R_rot,Ell[:,k])
    
                    ax[j][i].plot( u+Ell_rot[0] , v+Ell_rot[1],fillcol, ls=ls1[m], lw=1, label=r"%0.2f"%(CL[m]*100)+"\%")
                    ax[j][i].fill( u+Ell_rot[0] , v+Ell_rot[1],c=fillcol,alpha=fillalph )
    
                #######################################################################
    
                ax[j][i].set_xlim(u-5*sigx, u+5*sigx)
                ax[j][i].set_ylim(v-5*sigy, v+5*sigy)
    
                ax[j][i].vlines(u, v-5*sigy, v+5*sigy, color="slateblue", alpha=0.5)
                ax[j][i].hlines(v, u-5*sigx, u+5*sigx, color="slateblue", alpha=0.5)
    
        for i in range(len(C)):
            for j in range( i+1,len(C),1):
                ax[i][j].axis("off")
    
        for i in range(0, len(C)-1,1):
            for j in range(0,i+1,1):
                ax[i][j].set_xticklabels([])
    
            for j in range(1, i+2,1):
                ax[i+1][j].set_yticklabels([])
    
        ax[0][0].set_yticklabels([])
        ax[0][0].set_ylabel(" ")
        fig.tight_layout()
        if (legend is True):
            ax[1][0].legend(bbox_to_anchor=(2, 2), loc='center')
    
        if (label is not None):
            ax[0][0].legend(bbox_to_anchor=(2,0.2), loc="center")
    
        return fig, ax

    def get_all(self, data_cov, data_var,d_theta, mu,
            return_norm=False,Hartlap_correct=False,jk=False, names=None, CL=[0.683, 0.954, 0.9973], ls1=["-", "--", ":"],
            fillcol="mediumpurple",fillalph=0.2 , fig=None, ax=None, legend=True, label=None, sig_yval=0.8):
        """ Given the data and d_theta, calculates the data covariance matrix, model derivatives, Fisher matrix and plots the corner plots.
        inputs:
        data_cov: data for calculating the covariance matrix; shape [Nreal, m]
        data_var: data of the variation of the parameters; shape [N, m ,d] (where d=3 or 5)
        d_theta: array of values of parameters by which the plus-minus variations are performed. shape[N]
        mu: Parameter values at fiducial. shape[N]
        return_norm: returns the normalised variance matrix if set to True. Default is False.
        Hartlap_correct: the correction on inverse covariance matrix. Set to False.
        jk: True if the input data is from jack knife samples
        **kwargs: plotting, see plot_corner.
        
        outputs:
        Dictionary D with the following keys
        C: Data covariance matrix
        CI: inverse data covariance matrix
        F: Fisher matrix
        FI: Inverse fisher matrix
        dmdtheta: model derivatives
        NC: Normalised data covariance matrix if return_norm is True
        
        fig, ax: for the corner plots"""
        print("\nCalculating the data covariance and its inverse.")
        m=len(data_cov[0])
        D=self.calc_cov(data_cov, return_norm=return_norm, Hartlap_correct=Hartlap_correct, jk=jk)
        print("\n Covariance calculated.\n Calculating model derivatives.")

        N=len(d_theta)
        dmdtheta=np.zeros([N, m])
        for i in range(N):
            dmdtheta[i]=self.calc_deriv(data_var[i], d_theta[i])
        print("\nModel derivatives calculated.\nCalculating Fisher Matrix.")

        D["dmdtheta"]=dmdtheta
        CI=np.array(D["CI"])

        D1=self.calc_fisher(CI, dmdtheta)
        D["F"]=D1["F"]
        D["FI"]=D1["FI"]

        print("\nFisher matrix calculated. Plotting the corner plots")
        FI=np.array(D["FI"])
        fig, ax=self.plot_corner(FI, mu=mu, names=names, CL=CL,ls1=ls1, 
                fillcol=fillcol, fillalph=fillalph,fig= fig,ax= ax,legend= legend,label= label, sig_yval=sig_yval)

        return D, fig, ax

