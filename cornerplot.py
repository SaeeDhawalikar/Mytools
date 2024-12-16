import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/mnt/home/student/csaee/perl5/my_tools/')
import pretty


def plot_corner(C,mu, names=None, CL=[0.683, 0.954, 0.9973], ls1=["-", "--", ":"], 
        fillcol="mediumpurple",fillalph=0.2 ):
    """ Plots the corner plots given the inverse Fisher matrix C (or the covariance matrix of the parameters).
    Inputs:
    C: N x N covariance matrix
    mu: fiducial parameter values (array of length N)
    names: labels of the parameters (list of length N)
    CL: list of confidence levels to be plotted (0<CL<1). Default is [0.683, 0.954, 0.9973])
    ls1: line styles for each CL. Ensure len(ls1)=len(CL)
    fillcol: colour to fill the contours (default is "purple")
    fillalph: alpha of the fill (default is 0.2)
    Outputs the figure, and axes"""

    if (len(C)!=len(mu)):
        print("lenght of C does not match length of mu, exiting")
        return -1
    if (len(CL)!=len(ls1)):
        print("length of CL does not match length of line styles ls1, exiting")
        return -1
    CL=np.array(CL)
    alph=np.sqrt(-2*np.log(1-CL)) # scaling factor of ellipse

    fig, ax=plt.subplots(len(C),len(C), figsize=(15,15))
    for i in range(len(C)):
        sig=np.sqrt(C[i][i])
        x=np.linspace(-5*sig, 5*sig, 100)+mu[i]
        ax2=ax[i][i].twinx()
        y=np.exp(-(x-mu[i])**2/(2*sig**2))/np.sqrt(2*np.pi*sig**2)
        ax2.plot(x,y , "k")
        ax2.vlines(mu[i],0, np.max(y), color="slateblue")
        ax2.vlines(mu[i]+sig, 0, np.max(y), ls=":", color="k")
        ax2.vlines(mu[i]-sig, 0, np.max(y), ls=":", color="k")
        ax[-1][i].set_xlabel(names[i])
        ax[i][0].set_ylabel(names[i])
        ax2.set_yticklabels([])

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

                ax[j][i].plot( u+Ell_rot[0] , v+Ell_rot[1],"k", ls=ls1[m], lw=1, label=r"%0.2f"%(CL[m]*100)+"\%")
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
    ax[1][0].legend(bbox_to_anchor=(2, 2), loc='center')

    return fig, ax

