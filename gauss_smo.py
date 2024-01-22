import numpy as np
import pandas as pd
import sys
from astropy.io import fits
import smoothing_library as SL

#creates gaussian smoothed field from a gridded density

file="cic_512.fits"
ar=fits.open(file)[0].data
l=300 # boxsize
pix=512 # number of pixels

z=np.arange(l/(2*pix), l, l/pix) # pixel edges
ar[ar==0]=1e-10


ar=np.array(ar, dtype="<f4")
BoxSize=l
R       = l/pix 
grid    = ar.shape[0]
print(grid)
Filter  = 'Gaussian'
threads = 28

# compute FFT of the filter
W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)

# # smooth the field
field_smoothed = SL.field_smoothing(ar, W_k, threads)

hdu=fits.PrimaryHDU(field_smoothed)
hdu.writeto("sm_512.fits")
