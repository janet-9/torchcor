'''
@author: Shuang Qian
'''
import argparse
import numpy as np
from numpy import genfromtxt
import pandas as pd
import math
import glob
import random
import sys
import os.path, time 
import torchcor.electrodes.meshIO as meshIO
import torchcor.electrodes.findapexbase as meshUVC
import torchcor.electrodes.geometrical as geometrical

def find_electrodes(outputBasename,basenameLVmesh,basenameLVuvc,outfolder):
    #############
    ###This function is developed based on previous work:
    # Monaci, S.et al, 2021. Automated localization of focal ventricular tachycardia from simulated implanted device electrograms: 
    # A combined physics–AI approach. Frontiers in physiology, 12, p.682446. (10.3389/fphys.2021.682446)
    # To reuse, please also cite the above paper.
    
    #########################################
    # Loads in meshes and data
    ##########################################
    # Output basename
    #outputBasename = args.outputDir

    # Reads in mesh files for meshin need of electrodes
    #basenameLVmesh = args.source_msh
    pts_LV = meshIO.read_pts(basename=basenameLVmesh, file_pts=None)

    # UVC data
    #basenameLVuvc = args.uvcFolder
    Z_LV = meshUVC.read_uvcCoord(dataFilename=basenameLVuvc+'/COORDS_Z_LV.dat')
    RHO_LV = meshUVC.read_uvcCoord(dataFilename=basenameLVuvc+'/PHO_LV.dat')
    PHI_LV = meshUVC.read_uvcCoord(dataFilename=basenameLVuvc+'/PHI_LV.dat')

    #########################################
    # Finds location of apex, COG base and long axis for LV mesh
    #########################################
    # Locates apex point
    apexCoord_LV = meshUVC.find_apex(Z_LV,RHO_LV,pts_LV)

    # Locates base COG
    baseCOG_LV = meshUVC.find_baseCOG(Z_LV,RHO_LV,PHI_LV,pts_LV)

    # Computes LV long-axis
    longAxis_LV = (apexCoord_LV - baseCOG_LV)/np.linalg.norm(apexCoord_LV-baseCOG_LV)

    #########################################
    # Defines location of apex, COG base and long axis for torso mesh, as well as electrode set
    #########################################
    # Defines coordinates of apex in reference torso
    apexCoord_torso = np.array([263156.406250, 54942.023438, 197648.718750])

    # Defines COG of base in reference torso
    baseCOG_torso = np.array([223882.390625, 141031.4765626667, 210912.7552083333])

    # Defines long-axis of torso mesh
    longAxis_torso = (apexCoord_torso - baseCOG_torso)/np.linalg.norm(apexCoord_torso - baseCOG_torso)

    # Defines location electrode set to be rotated
    electrodes = np.array([[181597.4, 26021.5, 239365.5], [247209.1, 28483.71, 235889.9], [264670.7, 20444.59, 216705.8], [298350.8, 21707.97, 194372.1], [352309.2, 59260.08, 197485.6],  [410969.3, 170271.5, 213359.9], [31442.08, 103554.3, 348098.3], [394487, 113244.6, 352806.7], [36385.63, 108196.8, 5591.906], [376849.6, 91936.8, 7179.315]])

    #########################################
    # Performs initial translation to align bases
    #########################################
    t = baseCOG_torso - baseCOG_LV
    electrodes_t = electrodes - t
    baseCOG_torso_t = baseCOG_torso - t

    #########################################
    # Performs initial rotation to align long-axes
    #########################################
    # Performs rotation of long-axes
    R = geometrical.computeRotationMatrix(longAxis_LV,longAxis_torso)

    # Rotates various points in torso space
    electrodes_r = np.transpose(np.matmul(R,np.transpose(electrodes_t)))
    baseCOG_torso_r = np.transpose(np.matmul(R,np.transpose(baseCOG_torso_t)))

    # Align again with the COG of bases
    t2 = baseCOG_torso_r - baseCOG_LV
    electrodes_r = electrodes_r - t2
    baseCOG_torso_r = baseCOG_torso_r - t2

    #########################################
    # Defines short-axis in LV
    ######################################### 
    # Finds point in basal plane with phi=0
    base0phiPt_LV = meshUVC.find_base0phiPt(Z_LV, RHO_LV, PHI_LV, pts_LV)

    # Projects this point to be in same plane as COG_base point (both lying in plane normal to long axis)
    base0phiPt_LV_projected = base0phiPt_LV - np.dot(base0phiPt_LV - baseCOG_LV, longAxis_LV)*longAxis_LV

    # Defines short-axis vector to be between this projected point and COG_base point
    shortAxis_LV = base0phiPt_LV_projected - baseCOG_LV
    shortAxis_LV = shortAxis_LV/np.linalg.norm(shortAxis_LV)

    #########################################
    # Defines short-axis in torso
    ######################################### 
    # Defines point in basal plane with phi=0
    base0phiPt_torso = np.array([205388.765625, 124708.976562, 193192.031250])
    # Performs same transformations as done on all torso COG and electrode points above
    base0phiPt_torso_t = base0phiPt_torso - t
    base0phiPt_torso_r =  np.transpose(np.matmul(R,np.transpose(base0phiPt_torso_t)))
    base0phiPt_torso_r = base0phiPt_torso_r - t2

    # Projects this point to be in same plane as COG_base point (both lying in plane normal to long axis)
    base0phiPt_torso_projected = base0phiPt_torso_r - np.dot(base0phiPt_torso_r - baseCOG_LV, longAxis_LV)*longAxis_LV

    # Defines short-axis vector to be between this projected point and COG_base point
    shortAxis_torso = base0phiPt_torso_projected - baseCOG_LV
    shortAxis_torso = shortAxis_torso/np.linalg.norm(shortAxis_torso)

    #########################################
    # Rotates to align short-axes
    ######################################### 
    R_sa = geometrical.computeRotationMatrix(shortAxis_LV, shortAxis_torso)

    # Transforms relevant points (electrodes and COGs based on above rotation)
    electrodes_r2 = np.transpose(np.matmul(R_sa,np.transpose(electrodes_r)))
    baseCOG_torso_r2 = np.transpose(np.matmul(R_sa,np.transpose(baseCOG_torso_r)))

    # Aligns again the two bases
    t3 = baseCOG_torso_r2 - baseCOG_LV
    electrodes_r3 = electrodes_r2 - t3

    #########################################
    # Writes out files
    #########################################
    rotatedElectrodesFilename = outfolder + '/rotatedElectrodes'
    electrodes_r3_out = pd.DataFrame(electrodes_r3) 
    meshIO.write_pts(rotatedElectrodesFilename, electrodes_r3_out)
    meshIO.write_auxpts(rotatedElectrodesFilename, electrodes_r3_out)

# electrode_data = {'V1': phie_data[pts_electrodes[0], :],
#                       'V2': phie_data[pts_electrodes[1], :],
#                       'V3': phie_data[pts_electrodes[2], :],
#                       'V4': phie_data[pts_electrodes[3], :],
#                       'V5': phie_data[pts_electrodes[4], :],
#                       'V6': phie_data[pts_electrodes[5], :],
#                       'RA': phie_data[pts_electrodes[6], :],
#                       'LA': phie_data[pts_electrodes[7], :],                  
#                       'RL': phie_data[pts_electrodes[8], :],
#                       'LL': phie_data[pts_electrodes[9], :]}
