##This script is developed by Martin Bishop.
import pandas as pd
import glob
import numpy as np


#########################################
# Function to read uvc full pts file
#########################################
def read_uvcpts(basename=None, file_pts=None):
    # Function to read in mesh from basename

    if file_pts is None:
        file_pts = glob.glob(basename + '.pts')
        #if len(file_pts) > 1:
         #   raise ValueError('Too many matching .pts files')
        if len(file_pts) == 0:
            raise ValueError('No matching .pts files')
        file_pts = file_pts[0]

    # Read mesh files
    pts = pd.read_csv(file_pts, sep=' ', skiprows=1, header=None)
    print("Successfully read {}".format(file_pts))
    print('Mesh has',len(pts),'UVC nodes')

    return pts

#########################################
# Function to read individual mesh UVC coordinates
#########################################
def read_uvcCoord(dataFilename=None):
    # Function to read in mesh from basename
    
    if dataFilename is None:
        raise ValueError('No data to read')
        dataFilename = dataFilename[0]
    
    # Read mesh files
    data = pd.read_csv(dataFilename, sep=' ', skiprows=0, header=None).values.squeeze()
    print("Successfully read {}".format(dataFilename))

    return data

#########################################
# Function to compute centroids of UVC data
#########################################
def compute_uvc_centroids(elems=None, uvcs=None):

    uvc_n0 = np.array(uvcs.iloc[elems.iloc[:,0]])
    uvc_n1 = np.array(uvcs.iloc[elems.iloc[:,1]])
    uvc_n2 = np.array(uvcs.iloc[elems.iloc[:,2]])
    uvc_n3 = np.array(uvcs.iloc[elems.iloc[:,3]])
    
    mean_uvc = (uvc_n0 + uvc_n1 + uvc_n2 + uvc_n3)*0.25
    
    # Iterates over all elements performing continuity checks
    for i in range(0,len(elems)):
        
        # Checks V coordinate - defaults to LV
        if mean_uvc[i, 3] != 1:
            mean_uvc[i, 3] = -1
        
        # Checks RHO coordinate
        if mean_uvc[i, 1] > 1:
            mean_uvc[i, 1] = 1.0
            
        rhos = np.array([uvc_n0[i, 1],  uvc_n1[i, 1], uvc_n2[i, 1], uvc_n3[i, 1]])
        if (rhos > 0.9).sum() > 0:
            if (rhos < 0.1).sum() > 0:
                mean_uvc[i, 1] = 1.0
            
        # Checks PHI coordinate
        phis = np.array([uvc_n0[i, 2],  uvc_n1[i, 2], uvc_n2[i, 2], uvc_n3[i, 2]])
        if (phid > 2.8).sum() > 0:
            if (phis < -2.8).sum() > 0:
                mean_uvc[i, 2] = 3.14    
        
    # Converts to pd dataframe
    centroids = pd.DataFrame(mean_uvc)
        
    return centroids


#########################################
# Function to find apex
#########################################
def find_apex(Z=None, RHO=None, pts=None):
    
    apex_ball = np.intersect1d(np.where(Z<0.015)[0],np.where(RHO==1))
    
    apexPt = apex_ball[np.argmin(Z[apex_ball])]
    
    apex = np.array(pts.iloc[apexPt])
    
    return apex

#########################################
# Function to find COG of base
#########################################
def find_baseCOG(Z=None, RHO=None, PHI=None, pts=None):
    
    base_rim = np.intersect1d(np.where(Z>Z[np.argmax(Z)]*0.9)[0],np.where(RHO>RHO[np.argmax(RHO)]*0.6)[0])
    
    base180 = base_rim[np.argmin(np.abs(PHI[base_rim] + np.pi))]
    base300 = base_rim[np.argmin(np.abs(PHI[base_rim] + np.pi/3))]
    base60 = base_rim[np.argmin(np.abs(PHI[base_rim] - np.pi/3))]
    
    base180_coords = np.array(pts.iloc[base180])
    base300_coords = np.array(pts.iloc[base300])
    base60_coords = np.array(pts.iloc[base60])
    
    baseCOG = (base180_coords + base300_coords + base60_coords)/3
    
    return baseCOG

#########################################
# Function to find point on base rim
#########################################
def find_base0phiPt(Z=None, RHO=None, PHI=None, pts=None):
    
    base_rim = np.intersect1d(np.where(Z>Z[np.argmax(Z)]*0.9)[0],np.where(RHO>RHO[np.argmax(RHO)]*0.6)[0])

    base0 = base_rim[np.argmin(np.abs(PHI[base_rim]))]
    
    base0_coords = np.array(pts.iloc[base0])
    
    return base0_coords

