import numpy as np

class discretize(object):
    ''' functions to discretize/ grid data from N-body simulations'''
    
    def disc2d(pos, l, pixels):
        #coordinates of the positions of the points
        pos_x=pos[0]
        pos_y=pos[1]

        #fractional size of each pixel
        h=1/pixels

        #mid points of the pixels in the grid
        midx= l*np.array(np.linspace(h/2, 1-h/2, pixels))
        midy= l*np.array(np.linspace(h/2, 1-h/2, pixels))

        #grid of mid points of the pixels
        x_grid, y_grid= np.meshgrid(midx, midy)
        x_grid= x_grid.flatten()
        y_grid= y_grid.flatten()

        #creating an array to store the discretised data
        data=np.zeros((pixels, pixels), dtype=float) 

        #finding the nearest grid point corresponding to each particle
        for i in range(len(pos_x)):
            dis_x=np.abs(midx-pos_x[i])
            dis_y=np.abs(midy-pos_y[i])

            indx=int(np.min(np.where(dis_x==np.min(dis_x))[0]))  #nearest x=const plane
            indy=int(np.min(np.where(dis_y==np.min(dis_y))[0]))
            #np.min is used because there might be multiple nearest planes, in which case the one closest to the origin is chosen

            #nearest 3 neighbours
            if ((pos_x[i]-midx[indx])>0):
                indx1=indx+1
            else:
                indx1=indx-1

            if ((pos_y[i]-midy[indy])>0):
                indy1=indy+1
            else:
                indy1=indy-1

            if (indx1<0):
                indx1=len(midx)-1 #when the cell containing the particles is leftmost, its leftward cell is the rightmost
            if (indy1<0):
                indy1=len(midy)-1
            if (indx1>=len(midx)):
                indx1=0
            if (indy1>=len(midy)):
                indy1=0

            lpix=l/pixels
            dx=np.abs(pos_x[i]-midx[indx])/lpix
            dy=np.abs(pos_y[i]-midy[indy])/lpix

            tx=1-dx
            ty=1-dy

            #increasing the value at the neighbouring pixels
            data[indy][indx]=data[indy][indx]+tx*ty

            data[indy][indx1]=data[indy][indx1]+dx*ty
            data[indy1][indx]=data[indy1][indx]+tx*dy

            data[indy1][indx1]=data[indy1][indx1]+dx*dy

        return data    
    
    
    def discretize_CIC(pos,l, pixels):
        """ This function produces a discrete density field from positions of N particles.
        Inputs:
        pos: 3XN array giving the positions of N particles. The coordinates must be between 0 and l.
        l: length of the box in units Mpc/h
        pixels: number of pixels per side (grid size)

        Output:
        pixels X pixels X pixels array corresponding to the discrete density field.
        """
        #coordinates of the positions of the points
        pos_x=pos[0]%l
        pos_y=pos[1]%l
        pos_z=pos[2]%l

        #modulo operator is used so that position of particles is not larger than l


        #fractional size of each pixel
        h=1/pixels

        #mid points of the pixels in the grid
        midx= l*np.array(np.linspace(h/2, 1-h/2, pixels))
        midy= l*np.array(np.linspace(h/2, 1-h/2, pixels))
        midz= l*np.array(np.linspace(h/2, 1-h/2, pixels))

#         #grid of mid points of the pixels
#         x_grid, y_grid, z_grid= np.meshgrid(midx, midy, midz)
#         x_grid= x_grid.flatten()
#         y_grid= y_grid.flatten()
#         z_grid= z_grid.flatten()

        #creating an array to store the discretised data
        data=np.zeros((pixels, pixels, pixels), dtype=float) 

        #finding the nearest grid point corresponding to each particle
        for i in range(len(pos_x)):
            dis_x=np.abs(midx-pos_x[i])
            dis_y=np.abs(midy-pos_y[i])
            dis_z=np.abs(midz-pos_z[i])

            indx=int(np.min(np.where(dis_x==np.min(dis_x))[0]))  #nearest x=const plane
            indy=int(np.min(np.where(dis_y==np.min(dis_y))[0]))
            indz=int(np.min(np.where(dis_z==np.min(dis_z))[0]))
            #np.min is used because there might be multiple nearest planes, in which case the one closest to the origin is chosen

            #nearest 7 neighbours
            if ((pos_x[i]-midx[indx])>0):
                indx1=indx+1
            else:
                indx1=indx-1

            if ((pos_y[i]-midy[indy])>0):
                indy1=indy+1
            else:
                indy1=indy-1

            if ((pos_z[i]-midz[indz])>0):
                indz1=indz+1
            else:
                indz1=indz-1

            if (indx1<0):
                indx1=len(midx)-1
            if (indy1<0):
                indy1=len(midy)-1
            if (indz1<0):
                indz1=len(midz)-1
            if (indx1>=len(midx)):
                indx1=0
            if (indy1>= len(midy)):
                indy1=0
            if (indz1>=len(midz)):
                indz1=0

            lpix=l/pixels

            dx=np.abs(pos_x[i]-midx[indx])/lpix #distance to the centre of the nearest cell in units of width of pixel
            dy=np.abs(pos_y[i]-midy[indy])/lpix
            dz=np.abs(pos_z[i]-midz[indz])/lpix


            tx=1-dx
            ty=1-dy
            tz=1-dz
            #increasing the value at the neighbouring pixels
            data[indx][indy][indz]=data[indx][indy][indz]+tx*ty*tz

            data[indx][indy][indz1]=data[indx][indy][indz1]+tx*ty*dz
            data[indx][indy1][indz]=data[indx][indy1][indz]+tx*dy*tz
            data[indx1][indy][indz]=data[indx1][indy][indz]+dx*ty*tz

            data[indx][indy1][indz1]=data[indx][indy1][indz1]+tx*dy*dz
            data[indx1][indy][indz1]=data[indx1][indy][indz1]+dx*ty*dz
            data[indx1][indy1][indz]=data[indx1][indy1][indz]+dx*dy*tz

            data[indx1][indy1][indz1]=data[indx1][indy1][indz1]+dx*dy*dz

        return data    
    
    def discretize_NGP(pos,l, pixels):
        """ This function produces a discrete density field from positions of N particles.
        Inputs:
        pos: 3XN array giving the positions of N particles. The coordinates must be between 0 and l.
        l: length of the box in units Mpc/h
        pixels: number of pixels per side (grid size)

        Output:
        pixels X pixels X pixels array corresponding to the discrete density field.
        """
        #coordinates of the positions of the points
        pos_x=pos[0]%l
        pos_y=pos[1]%l
        pos_z=pos[2]%l

        #fractional size of each pixel
        h=1/pixels

        #mid points of the pixels in the grid
        midx= l*np.array(np.linspace(h/2, 1-h/2, pixels))
        midy= l*np.array(np.linspace(h/2, 1-h/2, pixels))
        midz= l*np.array(np.linspace(h/2, 1-h/2, pixels))

#         #grid of mid points of the pixels
#         x_grid, y_grid, z_grid= np.meshgrid(midx, midy, midz)
#         x_grid= x_grid.flatten()
#         y_grid= y_grid.flatten()
#         z_grid= z_grid.flatten()

        #creating an array to store the discretised data
        data=np.zeros((pixels, pixels, pixels), dtype=float) 

        #finding the nearest grid point corresponding to each particle
        for i in range(len(pos_x)):
            dis_x=np.abs(midx-pos_x[i])
            dis_y=np.abs(midy-pos_y[i])
            dis_z=np.abs(midz-pos_z[i])

            indx=int(np.min(np.where(dis_x==np.min(dis_x))[0]))  #nearest x=const plane
            indy=int(np.min(np.where(dis_y==np.min(dis_y))[0]))
            indz=int(np.min(np.where(dis_z==np.min(dis_z))[0]))

            #np.min is used because there might be multiple nearest planes, in which case the one closest to the origin is chosen


            #increasing the value at the pixels where the point lies
            data[indx][indy][indz]=data[indx][indy][indz]+1

        return data    
