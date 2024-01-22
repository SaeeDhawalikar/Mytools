import numpy as np

def plot_slice(data,L, axis,r0=None,vmin=-1, vmax=3, cmap="binary",
              xlim=None, ylim=None, zlim=None,ax=None, label=True):
    """plots the slice perpendicular to the given axis, and having r0 axis value
    data: a pix X pix X pix array of density
    L: boxsize in Mpc/h
    axis: axis perpendicular to the slice, takes values 0,1,2
    r0: position at slice
    vmin, vmax: limits for colorbar
    cmap: default is "binary"
    xlim, ylim, zlim: limits of axes in the plane of the slice
    ax: axis over which the plot is to be plotted. If none, plots a new figure
    label: gives the coordiante of the axis perpendicular to the slice. Default is True."""
    pix=len(data)
    l_pix=L/pix
    if (ax is None):
        fig, ax=plt.subplots(1,1,figsize=(10,10))
    if (axis==0):
        x_pix=int(r0/l_pix) #slice to be plotted
        im=ax.imshow(np.log10(np.transpose(data[x_pix,:,:])) ,origin="lower", extent=[0, L, 0, L],
                     vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_aspect(1)
        ax.set_xlabel(r"$y$ (Mpc/h)")
        ax.set_ylabel(r"$z$ (Mpc/h)")
        if (zlim is None):
            ax.set_xlim(0,L)
        else:
            ax.set_xlim(ylim[0], ylim[1])
        if (ylim is None):
            ax.set_ylim(0,L)
        else:
            ax.set_ylim(zlim[0], zlim[1])
        if (label is True):    
            ax.text(0,L+10*l_pix,"x=%0.2f Mpc/h"%r0)
        
    elif (axis==1):
        y_pix=int(r0/l_pix) #slice to be plotted
        im=ax.imshow(np.log10(data[:,y_pix,:]) ,origin="lower", extent=[0, L, 0, L],vmin=vmin, vmax=vmax,cmap=cmap)
        ax.set_aspect(1)
        ax.set_xlabel(r"$z$ (Mpc/h)")
        ax.set_ylabel(r"$x$ (Mpc/h)")
        if (zlim is None):
            ax.set_xlim(0,L)
        else:
            ax.set_xlim(zlim[0], zlim[1])
        if (xlim is None):
            ax.set_ylim(0,L)
        else:
            ax.set_ylim(xlim[0], xlim[1])
        if (label is True):      
            ax.text(0,L+10*l_pix,"y=%0.2f Mpc/h"%r0)
        
    elif (axis==2):
        z_pix=int(r0/l_pix) #slice to be plotted
        im=ax.imshow(np.log10(np.transpose(data[:,:,z_pix])) ,origin="lower", extent=[0, L, 0, L],
                     vmin=vmin, vmax=vmax,cmap=cmap)
        ax.set_aspect(1)
        ax.set_xlabel(r"$x$ (Mpc/h)")
        ax.set_ylabel(r"$y$ (Mpc/h)")
        if (ylim is None):
            ax.set_xlim(0,L)
        else:
            ax.set_xlim(xlim[0], xlim[1])
        if (xlim is None):
            ax.set_ylim(0,L)
        else:
            ax.set_ylim(ylim[0], ylim[1])
        if (label is True):      
            ax.text(0,L+10*l_pix,"z=%0.2f Mpc/h"%r0)
            
#     fig.colorbar(im, label=r"$\log_{10}(\rho/ \bar{\rho})$")
            
    return ax, im