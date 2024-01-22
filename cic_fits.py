import numpy as np
import pandas as pd
import sys
from astropy.io import fits
import MAS_library as MASL
import read_sims

# creates cic density field from the given simulation, and saves it as fits files

#parameters of the simulation
h=0.70000
O_m=0.27600
O_l=0.72400
z=0
l=300.00 # box size in Mpc/h
N_dm= 1024**3 #number of particles
argument=int(1) #the realization to be used

######################################################################################
grid    = 128  #the 3D field will have grid x grid x grid voxels
BoxSize = l
MAS = 'CIC'  #mass-assigment scheme
verbose = True   #print information on progress

#simulation output files
g=("/scratch/aseem/sims/su1024/delta0.0/r%d/snapshot_001.0"%argument, "/scratch/aseem/sims/su1024/delta0.0/r%d/snapshot_001.1"%argument,
   "/scratch/aseem/sims/su1024/delta0.0/r%d/snapshot_001.2"%argument,"/scratch/aseem/sims/su1024/delta0.0/r%d/snapshot_001.3"%argument,
   "/scratch/aseem/sims/su1024/delta0.0/r%d/snapshot_001.4"%argument,"/scratch/aseem/sims/su1024/delta0.0/r%d/snapshot_001.5"%argument,
   "/scratch/aseem/sims/su1024/delta0.0/r%d/snapshot_001.6"%argument,"/scratch/aseem/sims/su1024/delta0.0/r%d/snapshot_001.7"%argument)

gad=read_sims.gadget(g, l=l)
Pos=gad.get_pos()
Pos=(Pos-BoxSize/(2*grid))%BoxSize #so that the grid points are at the centers of cells
# define 3D density field
den = np.zeros((grid,grid,grid), dtype=np.float32)

# construct 3D density field
MASL.MA(Pos, den, BoxSize, MAS, verbose=verbose)

print(np.min(den), np.max(den))
# at this point, delta contains the effective number of particles in each voxel
# now compute overdensity and density constrast
den /= np.mean(den, dtype=np.float64)
print(np.min(den), np.max(den))

print("saving file")
hdu=fits.PrimaryHDU(den)
hdu.writeto("cic_%d.fits"%grid)
