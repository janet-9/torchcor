import pandas as pd
import glob

############################
##
#This script is written by Martin Bishop.


#########################################
# Function to read pts file
#########################################
def read_pts(basename=None, file_pts=None):
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
    print('Mesh has',len(pts),'nodes')

    return pts


#########################################
# Function to read elems file
#########################################
def read_elems(basename=None, file_elem=None):
    # Function to read in mesh from basename

    if file_elem is None:
        file_elem = glob.glob(basename + '.elem')
        if len(file_elem) > 1:
            raise ValueError('Too many matching .elem files')
        elif len(file_elem) == 0:
            raise ValueError('No matching .elem files')
        file_elem = file_elem[0]

    # Read mesh files
    elem = pd.read_csv(file_elem, sep=' ', skiprows=1, usecols=(1, 2, 3, 4, 5), header=None)
    print("Successfully read {}".format(file_elem))
    print('Mesh has',len(elem),'elements')
    return elem

#########################################
# Function to write element file
#########################################
def write_elems(elemFilename=None, elem=None, shapes=None):
    # Write elem

    # Ensure *something* is being written!
    assert ((elem is not None)), "No data given to write to file."

    ######################
    # Writes-out elems file
    ######################
    # If we haven't defined a shape for our elements, set to be tets
    if shapes is None:
        shapes = 'Tt'

    if elem is not None:
        with open(elemFilename + '.elem', 'w') as pFile:
            pFile.write('{}\n'.format(len(elem)))
        elem.insert(loc=0, value=shapes, column=0, allow_duplicates=True)
        elem.to_csv(elemFilename + '.elem', sep=' ', header=False, index=False, mode='a')
        print("elem data written to file {}.".format(elemFilename + '.elem'))
        del elem[0]  # Remove added column to prevent cross-talk problems later

    return None



#########################################
# Function to write pts file
#########################################
def write_pts(ptsFilename=None, pts=None):
    # Ensure *something* is being written!
    assert ((pts is not None)), "No data given to write to file."

    precision_pts = '%.12g'

    ######################
    # Writes-out pts file
    ######################
    if pts is not None:
        with open(ptsFilename + '.pts', 'w') as pFile:
            pFile.write('{}\n'.format(len(pts)))
        pts.to_csv(ptsFilename + '.pts', sep=' ', header=False, index=False, mode='a', float_format=precision_pts)
        print("pts data written to file {}.".format(ptsFilename + '.pts'))

    return None
        

#########################################
# Function to write auxgrid pts file
#########################################
def write_auxpts(auxptsFilename=None, pts=None):
    # Ensure *something* is being written!
    assert ((pts is not None)), "No data given to write to file."

    precision_pts = '%.12g'

    ######################
    # Writes-out pts file
    ######################
    if pts is not None:
        with open(auxptsFilename + '.pts_t', 'w') as pFile:
            pFile.write('{}\n'.format(len(pts)))
            pFile.write("1\n")
        pts.to_csv(auxptsFilename + '.pts_t', sep=' ', header=False, index=False, mode='a', float_format=precision_pts)
        print("pts data written to file {}.".format(auxptsFilename + '.pts_t'))

    return None

#########################################
# Function to write out vtx file
#########################################
def write_vtx_File(vtxFilename=None, vtx=None):
    # Ensure *something* is being written!
    assert ((vtx is not None)), "No data given to write to file."
    
    if vtx is not None:
        #vtx.to_csv(vtxFilename + '.vtx', header=False, index=False, mode='a')
        dFile = open(vtxFilename+ '.vtx', '+w') 
        #dFile.write("1\n")
        dFile.write("%i\n" %len(vtx))
        dFile.write("extra\n")
        for i in range(len(vtx)):
            dFile.write("%i\n" %vtx.loc[i])
        #vtx.to_csv(vtxFilename + '.vtx', header=False, index=False, mode='a')
        dFile.close()   
        print("vtx data written to file {}.".format(vtxFilename + '.vtx'))
    return None


