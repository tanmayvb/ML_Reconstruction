import numpy as np

#--------------------------------------------
#Run 
#uv run python run_create_mask.py
#--------------------------------------------

def create_mask(shape, points, sigma_z=2, sigma_xy=6):
    mask = np.zeros(shape, dtype=np.float32)

    for (zc, yc, xc) in points:
        for z in range(max(0,int(zc-3*sigma_z)), min(shape[0],int(zc+3*sigma_z))):
            for y in range(max(0,int(yc-3*sigma_xy)), min(shape[1],int(yc+3*sigma_xy))):
                for x in range(max(0,int(xc-3*sigma_xy)), min(shape[2],int(xc+3*sigma_xy))):

                    dz = (z-zc)**2/(2*sigma_z**2)
                    dy = (y-yc)**2/(2*sigma_xy**2)
                    dx = (x-xc)**2/(2*sigma_xy**2)

                    val = np.exp(-(dz+dy+dx))
                    mask[z,y,x] = max(mask[z,y,x], val)

    mask = mask / (mask.max() + 1e-8) #Normalize mask added new
    mask = (mask > 0.2).astype(np.float32)
    return mask
