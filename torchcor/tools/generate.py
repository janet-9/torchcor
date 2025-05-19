import os
import numpy as np
import vtk
import pickle
from vtk.util.numpy_support import  numpy_to_vtk, vtk_to_numpy
from pathlib import Path


########################################################################################
#######                                                                          #######
#######                           VTK                                            #######
#######                                                                          #######
########################################################################################
def readvtk(filename: str) -> vtk.vtkPolyData:
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()    
    reader.Update()
    return reader.GetOutput()

def writevtk(polydata :vtk.vtkPolyData,filename : str,binary : bool = True):
    writer = vtk.vtkPolyDataWriter()
    if(vtk.VTK_MAJOR_VERSION<6):
      writer.SetInput(polydata)
    else:
      writer.SetInputData(polydata)
    if(binary):
      writer.SetFileTypeToBinary()
    else:
      writer.SetFileTypeToASCII()
    writer.SetFileName(filename)
    writer.Write()

def getPointArrayNames(polydata : vtk.vtkPolyData) -> list[str]:
    '''Extracts all the array names '''
    arraylist=[]
    for iv in range(0,polydata.GetPointData().GetNumberOfArrays()):
        arraylist.append(polydata.GetPointData().GetArrayName(iv))
    return arraylist 

def getCellArrayNames(polydata : vtk.vtkPolyData) -> list[str]:
    '''Extracts all the array names '''
    arraylist=[]
    for iv in range(0,polydata.GetCellData().GetNumberOfArrays()):
        arraylist.append(polydata.GetCellData().GetArrayName(iv))
    return arraylist 

def add_PointData_array(polydata : vtk.vtkPolyData, arrayname : str, initval : float = 0.0) -> vtk.vtkDoubleArray:
    ''' adds a new point data array to polydata with name arrayname '''
    newarray = vtk.vtkDoubleArray()
    newarray.SetName(arrayname)
    newarray.SetNumberOfTuples(polydata.GetNumberOfPoints())
    newarray.FillComponent(0, initval)
    polydata.GetPointData().AddArray(newarray)
    return(newarray)

def add_CellData_array(polydata: vtk.vtkPolyData, arrayname: str, initval: float = 0.0) -> vtk.vtkDoubleArray:
    ''' adds a new point data array to polydata with name arrayname '''
    newarray = vtk.vtkDoubleArray()
    newarray.SetName(arrayname)
    newarray.SetNumberOfTuples(polydata.GetNumberOfCells())
    newarray.FillComponent(0, initval)
    polydata.GetCellData().AddArray(newarray)
    return(newarray)

def add_pointData_array_numpy(name : str,polydata: vtk.vtkPolyData, numpyArray : np.ndarray,overwrite : bool =False) -> vtk.vtkFloatArray:
    if polydata.GetPointData().GetArray(name):
        if overwrite:
            polydata.GetPointData().RemoveArray(name)
        else:    
            return(polydata.GetPointData().GetArray(name))
    newarray = numpy_to_vtk(numpyArray.astype(np.float32),deep=1)
    newarray.SetName('{}'.format(name) )
    polydata.GetPointData().AddArray(newarray)
    return(polydata.GetPointData().GetArray(name))


def add_CellData_array_numpy(name : str,polydata: vtk.vtkPolyData, numpyArray : np.ndarray,overwrite : bool =False) -> vtk.vtkFloatArray:
    if polydata.GetCellData().GetArray(name):
        if overwrite:
            polydata.GetCellData().RemoveArray(name)
        else:
            return(polydata.GetCellData().GetArray(name))
    newarray = numpy_to_vtk(numpyArray.astype(np.float32),deep=1)
    newarray.SetName('{}'.format(name) )
    polydata.GetCellData().AddArray(newarray)
    return(polydata.GetCellData().GetArray(name))


## Note: using NT criterion (fibrosis for IIR>1.22):
# Region 1 (normal tissue) is formed by IDs 1,2,3
# Region 2 (fibrotic tissue) is formed by IDs 4,5,6
def assign_region_label(IIRv: float) -> int:
    if (IIRv<=0.9):
        return(1)
    elif (IIRv<=1.1):
        return(2)
    elif (IIRv<=1.22):
        return(3)
    elif (IIRv<=1.4):
        return(4)
    elif (IIRv<=1.6):
        return(5)
    else:
        return(6)      

def extract_mesh(polydata : vtk.vtkPolyData, extract_data: bool =True) -> dict[np.ndarray]:
    p_radius   = 2.0*1000 #(2 mm in microns)
    nPts   = polydata.GetNumberOfPoints()
    nTri   = polydata.GetNumberOfCells()
    Pts    = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float32)
    Tria   = np.zeros(shape=(nTri,4),dtype=int)
    Fibers = np.zeros(shape=(nTri,3),dtype=np.float32)
    IIR_array = polydata.GetPointData().GetArray('IIR')
    fib_array = polydata.GetCellData().GetArray('fiber_endo')    
    for cellID in range(nTri):
        elem = polydata.GetCell(cellID)
        pID0 = int(elem.GetPointId(0))
        pID1 = int(elem.GetPointId(1))
        pID2 = int(elem.GetPointId(2))
        IIR0 = IIR_array.GetValue(pID0)
        IIR1 = IIR_array.GetValue(pID1)
        IIR2 = IIR_array.GetValue(pID2)
        IIR  = (IIR0+IIR1+IIR2)/3.0
        rID  = int(assign_region_label(IIR))
        fx   = fib_array.GetComponent(cellID,0)
        fy   = fib_array.GetComponent(cellID,1)
        fz   = fib_array.GetComponent(cellID,2)
        fF   = np.sqrt(fx*fx + fy*fy + fz*fz)
        Tria[cellID,:]   = np.array([pID0,pID1,pID2,rID],dtype=int)
        Fibers[cellID,:] = np.array([fx/fF,fy/fF,fz/fF],dtype=np.float32)
    mesh0 = {'Pts': Pts,
            'Tria': Tria,
            'Fibers':Fibers}

    if extract_data:
        UAC1    = vtk_to_numpy(polydata.GetPointData().GetArray('UAC1') )
        UAC2    = vtk_to_numpy(polydata.GetPointData().GetArray('UAC2'))
        IUAC    = np.logical_and(np.logical_and(UAC1>=0.5, UAC1<=0.7 ),np.logical_and(UAC2>=0.8, UAC2<=0.9 ))
        #
        c0      = Pts[IUAC,:].mean(axis=0)
        IUAC    = np.linalg.norm(Pts-c0,axis=1)<=p_radius
        #
        stim    = np.where(IUAC)[0].astype(int)
        mesh0['stim'] = stim
        if polydata.GetPointData().GetArray('inilats'):
        	iniLATS = vtk_to_numpy(polydata.GetPointData().GetArray('inilats'))
        	mesh0['iniLATS'] = iniLATS
        if polydata.GetPointData().GetArray('PV_pacings'):
        	PV_pacings = vtk_to_numpy(polydata.GetPointData().GetArray('PV_pacings'))
        	mesh0['PV_pacings'] = PV_pacings
    return(mesh0)
    
    
    
########################################################################################
#######                                                                          #######
#######                           CARP                                           #######
#######                                                                          #######
########################################################################################

def write_carp_nodes(mesh0 : dict[np.ndarray], prefixname : str,convert : float= 1.0):
    ''' writes the .pts file for carp meshes '''
    points = mesh0['Pts']
    with open('{0}.pts'.format(prefixname), 'w') as fp:
      fp.write('{0}\n'.format(points.shape[0]) )
      for pt in points:
          fp.write('{0} {1} {2}\n'.format(convert*pt[0],convert*pt[1],convert*pt[2]) )


def write_carp_triangles(mesh0 : dict[np.ndarray], prefixname : str):
    ''' writes the .elem file for carp meshes '''
    Tria   = mesh0['Tria']
    nCells = Tria.shape[0]
    with open('{0}.elem'.format(prefixname), 'w') as fe:
        fe.write('{0}\n'.format(nCells) )
        for Tr in Tria:
            fe.write('Tr {0} {1} {2} {3}\n'.format(Tr[0],Tr[1],Tr[2],Tr[3]) )


def write_carp_fibres(mesh0 : dict[np.ndarray],prefixname : str):
    Fibers = mesh0['Fibers']
    with open('{0}.lon'.format(prefixname), 'w') as fe:
        fe.write('{0:d}\n'.format(1))
        for fib in Fibers:
            fe.write('{0:5.4f} {1:5.4f} {2:5.4f}\n'.format(fib[0],fib[1],fib[2]))


def write_carp_stimulus_sets(mesh0: dict[np.ndarray],stimfname):
    if 'stim' in mesh0.keys():
        stimregion = mesh0['stim']
        fname = '{0}.vtx'.format(stimfname)       
        with open(fname,'w') as f:
            f.write('{0:d}\nintra\n'.format(stimregion.shape[0]) )
            for jl in stimregion:
                f.write('{0:d}\n'.format(jl))
    if 'PV_pacings' in mesh0.keys(): 
        PV_pacings = mesh0['PV_pacings'].astype(int)
        for jj,vein_name in enumerate(['LSPV','LIPV','RSPV','RIPV','LAA','roof']):
            stimregion = np.where(PV_pacings==(1+jj))[0]
            fname = '{}_{}.vtx'.format(stimfname,vein_name)
            with open(fname,'w') as f:
                f.write('{0:d}\nintra\n'.format(stimregion.shape[0]) )
                for jl in stimregion:
                    f.write('{0:d}\n'.format(jl))


def write_carp_LATs(mesh0: dict[np.ndarray],latsfname):
    ''' used to put a 0 LATs to all the cell to use prepacing'''
    if 'iniLATS' in mesh0.keys():
        iniLATS = mesh0['iniLATS']
        with open('{0}_iniLATS.dat'.format(latsfname),'w') as fp:
           for lat in iniLATS:
                 fp.write('{0:5.4f}\n'.format(lat))
    points = mesh0['Pts']
    with open('{0}_iniLATS0.dat'.format(latsfname),'w') as fp:
       for p in range(points.shape[0]):
             fp.write('{0:5.4f}\n'.format(0))

def write_carp_files(mesh0: dict[np.ndarray] ,outputdir: str, prefix: str, write_lats : bool=True):
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    prefixname = os.path.join(outputdir,prefix)
    stimfname  = os.path.join(outputdir,prefix)
    latsfname  = os.path.join(outputdir,prefix)
    write_carp_nodes(mesh0,prefixname)
    write_carp_triangles(mesh0,prefixname)
    write_carp_fibres(mesh0,prefixname)
    write_carp_stimulus_sets(mesh0,stimfname)
    if write_lats:
        write_carp_LATs(mesh0,latsfname)    
    
    
    
    
    
    
      
    
    


def adaptiveSubdiv(polydata,edgeLen=0.2):
    adapt = vtk.vtkAdaptiveSubdivisionFilter()
    if(vtk.VTK_MAJOR_VERSION<6):
        adapt.SetInput(polydata)
    else:
        adapt.SetInputData(polydata)
    adapt.SetMaximumEdgeLength(edgeLen)
    adapt.SetMaximumTriangleArea(0.5*edgeLen*edgeLen)
    adapt.Update()
    polydata = cleanpolydata(adapt.GetOutput())
    return(polydata)


def compute_mesh_quality(polydata):
    #see here for qualities: https://vtk.org/Wiki/images/6/6b/VerdictManual-revA.pdf
    qfilt= vtk.vtkMeshQuality()
    if vtk.VTK_MAJOR_VERSION <= 6:
       qfilt.SetInput(polydata)
    else:
       qfilt.SetInputData(polydata)   
    qfilt.SetTriangleQualityMeasureToArea()
    #qfilt.SaveCellQualityOn() #default
    qfilt.SaveCellQualityOff() #stores only final statistics
    qfilt.Update() #creates vtkDataSet
    #quality = qfilt.GetOutput().GetCellData().GetArray("Quality")
    return(qfilt.GetOutput())

def print_mesh_quality(polydata,reducedprint=False):
        npt          = polydata.GetNumberOfPoints()
        nCell        = polydata.GetNumberOfCells()
        qfdata       = compute_mesh_quality(polydata)
        qualityArray = qfdata.GetFieldData().GetArray('Mesh Triangle Quality')
        #minimum, average, maximum, and unbiased variance
        quality = {'min': np.sqrt(qualityArray.GetValue(0)),
                   'avg':  np.sqrt(qualityArray.GetValue(1)),
                   'max':  np.sqrt(qualityArray.GetValue(2)),
                   'std':  np.sqrt(np.sqrt(qualityArray.GetValue(3))),
                   'var':  np.sqrt(qualityArray.GetValue(3))
                 }
        if reducedprint:
            avgE = quality['avg']
            stdE = quality['std']
            maxE = quality['max']
            print('npt={}\tnel={}\tavg={:4.2f}\tstd={:4.2f}\tmax={:4.2f}'.format(npt,nCell,avgE,stdE,maxE),flush=True )
        else:
            print('npt={},nel={}'.format(npt,nCell),flush=True )
            for key in quality.keys():
                print('L ({}) = {} [microns]'.format(key,quality[key]),flush=True)
            print('----------------------------------',flush=True)



### clean the topolgy removing unused point/cells
def cleanpolydata(polydata):
    cleaner = vtk.vtkCleanPolyData()
    a=int(vtk.vtkVersion.GetVTKVersion()[0])
    if(a<6):
      cleaner.SetInput(polydata)
    else:
      cleaner.SetInputData(polydata)
    cleaner.Update()
    return cleaner.GetOutput()


def generate_PV_pacing_sites(polydata: vtk.vtkPolyData) -> vtk.vtkPolyData :
    # physical coordinates (in microns)
    Pts = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float32)
    # UACs
    X   = vtk_to_numpy(polydata.GetPointData().GetArray('UAC1'))
    Y   = vtk_to_numpy(polydata.GetPointData().GetArray('UAC2'))
    PV_pacings = np.zeros(X.shape,dtype=np.float32)
    p_radius   = 2.0*1000 #(2 mm in microns)    
    # LSPV:  .86-.88    .65-.75
    I  = np.logical_and( np.logical_and((X>=0.86),(X<=0.88)  ), np.logical_and((Y>=0.65),(Y<=0.75) ) )
    c0 = Pts[I,:].mean(axis=0)
    PV_pacings[np.linalg.norm(Pts-c0,axis=1)<=p_radius] = 1.0
    #PV_pacings[I] = 1.0
    
    # LIPV:  .85-.86    .25-.26
    I  = np.logical_and( np.logical_and((X>=0.85),(X<=0.86)  ), np.logical_and((Y>=0.25),(Y<=0.26) ) )
    c0 = Pts[I,:].mean(axis=0)
    PV_pacings[np.linalg.norm(Pts-c0,axis=1)<=p_radius] = 2.0
    #PV_pacings[I] = 2.0
    
    # RSPV:  .05-.08    .60-.63
    I  = np.logical_and( np.logical_and((X>=0.05),(X<=0.08)  ), np.logical_and((Y>=0.60),(Y<=0.63) ) )
    c0 = Pts[I,:].mean(axis=0)
    PV_pacings[np.linalg.norm(Pts-c0,axis=1)<=p_radius] = 3.0
    #PV_pacings[I] = 3.0
    
    # RIPV:  .12-.13    .25-.26
    I  = np.logical_and( np.logical_and((X>=0.12),(X<=0.13)  ), np.logical_and((Y>=0.25),(Y<=0.26) ) )
    c0 = Pts[I,:].mean(axis=0)
    PV_pacings[np.linalg.norm(Pts-c0,axis=1)<=p_radius] = 4.0
    #PV_pacings[I] = 4.0
    
    # LAA:  .9-.91    .9 - .91
    I = np.logical_and( np.logical_and((X>=0.9),(X<=0.91)  ), np.logical_and((Y>=0.9),(Y<=0.91) ) )
    c0 = Pts[I,:].mean(axis=0)
    PV_pacings[np.linalg.norm(Pts-c0,axis=1)<=p_radius] = 5.0
    
    # ROOF:  .48-.49    .48 - .49
    I = np.logical_and( np.logical_and((X>=0.48),(X<=0.49)  ), np.logical_and((Y>=0.48),(Y<=0.49) ) )
    c0 = Pts[I,:].mean(axis=0)
    PV_pacings[np.linalg.norm(Pts-c0,axis=1)<=p_radius] = 6.0

    PV_pacings    = add_pointData_array_numpy('PV_pacings',polydata, PV_pacings)
    
    return(polydata)


def generate_spiral_initialisation(polydata: vtk.vtkPolyData) -> vtk.vtkPolyData :
    X = vtk_to_numpy(polydata.GetPointData().GetArray('UAC1'))
    Y = vtk_to_numpy(polydata.GetPointData().GetArray('UAC2'))
    X = 2.0*X - 1.0
    Y = 2.0*Y - 1.0
    
    # 1st spiral
    xcen     = 0.5
    ycen     = 0.5
    Xshift   = X - xcen
    Yshift   = Y - ycen
    d1       = np.sqrt(Xshift**2 + Yshift**2)
    a1       = np.arctan2(Yshift, Xshift)
    kappa    = 1
    omega    = -1
    D_spiral = 1
    t        = ( -2.0 * np.pi / (D_spiral * omega) * d1 + 1.0 / omega * a1 + 2.0 * np.pi * kappa / omega  )

    # 2nd spiral
    xcen2    = -0.5
    ycen2    = 0.5
    Xshift2  = X - xcen2
    Yshift2  = Y - ycen2
    d2       = np.sqrt(Xshift2**2 + Yshift2**2)
    a2       = np.arctan2(Yshift2, Xshift2)
    kappa    = 1
    omega    = -1
    D_spiral = -1
    time2    = (-2.0 * np.pi / (D_spiral * omega) * d2 + 1.0 / omega * a2 + 2.0 * np.pi * kappa / omega  )
    time2       = -time2 - 14
    time        = t
    TimeAll     = time
    TimeAll_new = np.where(time > time2, time2, TimeAll)
    TimeAll     = TimeAll_new

    # 3rd spiral
    xcen     = -0.5
    ycen     = -0.5
    Xshift   = X - xcen
    Yshift   = Y - ycen
    d1       = np.sqrt(Xshift**2 + Yshift**2)
    a1       = np.arctan2(Yshift, Xshift)
    kappa    = 1
    omega    = -1
    D_spiral = 1    
    t        = (-2.0 * np.pi / (D_spiral * omega) * d1 + 1.0 / omega * a1 + 2.0 * np.pi * kappa / omega  )
    TimeAll_new = np.where(TimeAll > t, t, TimeAll)
    TimeAll     = TimeAll_new

    # 4th spiral
    xcen2    = 0.5
    ycen2    = -0.5
    Xshift2  = X - xcen2
    Yshift2  = Y - ycen2
    d2       = np.sqrt(Xshift2**2 + Yshift2**2)
    a2       = np.arctan2(Yshift2, Xshift2)
    kappa    = 1
    omega    = -1    
    D_spiral = -1
    time2    = ( -2.0 * np.pi / (D_spiral * omega) *d2 + 1.0 / omega * a2 + 2.0 * np.pi * kappa / omega  )
    time2    = -time2 - 14.0
    TimeAll_new = np.where(TimeAll > time2, time2, TimeAll)
    TimeAll     = TimeAll_new
    TT          = TimeAll - np.min(TimeAll)
    TT          = TT * 34.0
    inilats     = add_pointData_array_numpy('inilats',polydata, TT)     
    return(polydata)
    
    
# CAROLINE DATASET IS E.G.:
#Sum: 1.87056e+08, Mean: 341.175, Stddev: 58.9917, Min: 3.70679, Max: 636.258

def get_uac_iir(polydata):
    X   = vtk_to_numpy(polydata.GetPointData().GetArray('UAC1'))
    Y   = vtk_to_numpy(polydata.GetPointData().GetArray('UAC2'))

    IIR_array = polydata.GetPointData().GetArray('IIR')

    return X, Y, IIR_array

def generate_simulation_data(filename,INPUTDIR,OUTDIR,CASEDIRNAME,refine=False):
    if filename[-4:]=='.vtk':
        filename = filename[:-4]
    FNAME = os.path.join(INPUTDIR,filename)
    surface = readvtk('{}.vtk'.format(FNAME ))
    surface = cleanpolydata(surface)
    print('mesh quality')
    print_mesh_quality(surface,reducedprint=refine)
    outputdir = os.path.join(OUTDIR,CASEDIRNAME)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    if refine:
        #print('refining case {}'.format(CASEDIRNAME) )
        surface = adaptiveSubdiv(surface,edgeLen=400)    # -------------------- 
        #print('New mesh quality:',flush=True)
        print_mesh_quality(surface,reducedprint=True)
        print('----------------------------------',flush=True)
    surface = generate_spiral_initialisation(surface)
    surface = generate_PV_pacing_sites(surface)
    X, Y, IIR = get_uac_iir(surface)
    uacpath = Path(OUTDIR) / CASEDIRNAME / "UAC_IIR.npz"
    np.savez(uacpath, UAC1=X, UAC2=Y, IIR=IIR)

    mesh0   = extract_mesh(surface)
    write_carp_files(mesh0,outputdir,CASEDIRNAME)
    surface = classify_surface(mesh0,surface)
    VTKdir = os.path.join(OUTDIR,'VTK')
    if not os.path.exists(VTKdir):
        os.makedirs(VTKdir)
    writevtk(surface,os.path.join(VTKdir,'{}.vtk'.format(CASEDIRNAME) )  )



def generate_pacingsites_only(filename,INPUTDIR,OUTDIR,CASEDIRNAME):
    if filename[-4:]=='.vtk':
        filename = filename[:-4]
    VTKdir  = os.path.join(OUTDIR,'VTK')
    FNAME   = os.path.join(VTKdir,'{}'.format(CASEDIRNAME) ) 
    surface = readvtk('{}.vtk'.format(FNAME ))
    # remove the existing array with the pacings
    if surface.GetPointData().GetArray('PV_pacings'):
        surface.GetPointData().RemoveArray('PV_pacings')
    surface = generate_PV_pacing_sites(surface)
    mesh0   = extract_mesh(surface)
    outputdir = os.path.join(OUTDIR,CASEDIRNAME)
    stimfname  = os.path.join(outputdir,CASEDIRNAME)
    write_carp_stimulus_sets(mesh0,stimfname)
    writevtk(surface,os.path.join(VTKdir,'{}.vtk'.format(CASEDIRNAME) )  )





def compute_surface_percentages(filename,INPUTDIR):
    if filename[-4:]=='.vtk':
        filename = filename[:-4]
    FNAME = os.path.join(INPUTDIR,filename)
    surface = readvtk('{}.vtk'.format(FNAME ))
    surface = cleanpolydata(surface)
    mesh0 = extract_mesh(surface,extract_data=False)
    Tria   = mesh0['Tria']
    Points = mesh0['Pts']
    surfAR = {'Total':0}
    for jj in range(6):
        surfAR[1+jj]=0.0
    for jElem,Elem in enumerate(Tria):
        ID = Elem[-1]
        l1 = Points[Elem[1],:]-Points[Elem[0],:]
        l2 = Points[Elem[2],:]-Points[Elem[0],:]
        Area = 0.5*np.linalg.norm(np.cross(l1,l2))
        surfAR[ID]+=Area
        surfAR['Total']+=Area
    return(surfAR)


def classify_surface(mesh0,polydata):
    TriaID   = mesh0['Tria'][:,-1]
    classification_array = add_CellData_array_numpy('regionID',polydata, TriaID)
    return(polydata)



    
if __name__=='__main__':
    REFINEFLAG = True
    BASEDIR  = "/data/Bei/" # os.getcwd() 
    INPUTDIR = os.path.join(BASEDIR,'5801337')
    if REFINEFLAG:
        OUTDIR   = os.path.join(BASEDIR,'meshes_refined')
    else:
        OUTDIR   = os.path.join(BASEDIR,'meshes')
    MAPFNAME = os.path.join(OUTDIR,'ID_to_NAME_MAP.pkl')    
    if not os.path.exists(OUTDIR):
            os.makedirs(OUTDIR)
    print(INPUTDIR)
    if not os.path.isfile(MAPFNAME):
        case_map = dict()
        for icase,CASENAME in enumerate(os.listdir(INPUTDIR)):
            case_map[1+icase]=CASENAME                
        with open(os.path.join(OUTDIR,'ID_to_NAME_MAP.pkl'),'wb') as  fout:
            pickle.dump(case_map,fout)
    else:      
        case_map=pickle.load(open(MAPFNAME,'rb'))
    
    NCASES = len(case_map.keys())
    PROPERTIES = np.zeros(shape=(NCASES,7))
    
    for JCASE,(icase,CASENAME) in enumerate(case_map.items()):
        CASEDIRNAME = 'Case_{}'.format(icase)
        print('{}'.format(CASEDIRNAME),flush=True)
        #generate_pacingsites_only(CASENAME,INPUTDIR,OUTDIR,CASEDIRNAME)
        generate_simulation_data(CASENAME,INPUTDIR,OUTDIR,CASEDIRNAME,refine=REFINEFLAG)
        surface_statistics = compute_surface_percentages(CASENAME,INPUTDIR)
        for JJ in range(6):
            PROPERTIES[JCASE,JJ] = surface_statistics[1+JJ]
        PROPERTIES[JCASE,6] = surface_statistics['Total']
        
        outputdir = os.path.join(OUTDIR,CASEDIRNAME)
        with open(os.path.join(outputdir,'surf_properties_areas.pkl'),'wb') as  fs:
            pickle.dump(surface_statistics,fs)
    np.save(os.path.join(OUTDIR,'tissue_properties.npy'),PROPERTIES)        
    




