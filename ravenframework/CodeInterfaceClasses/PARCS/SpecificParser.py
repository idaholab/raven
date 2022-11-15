"""
Created on Aug. 10th, 2022
@author: khnguy22
comment: Specific parser for PRACS interface
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import os
from xml.etree import ElementTree as ET
class DataParser():
  """
    Parse the data in RAVEN input in to PARCS input
    and be saved for the perturbed input generator in Perturbed parser
    Note: this data parser is only called once
  """
  def __init__(self, inputFile):
    """
      Constructor.
      @ In, inputFiles, string, xml PARCS parameters file
      @ In, workingDir, string, absolute path to the working directory
      @ Out, None
    """
    self.inputFile = inputFile
    self.getParameters()
  def getParameters(self):
    """
      Get required parameters from xml file for generating
      Constructor.
      @ In, None
      @ Out, None
    """
    fullFile = os.path.join(self.inputFile)
    dorm = ET.parse(fullFile)
    root = dorm.getroot()
    self.THFlag = root.find('THFlag').text.strip()
    self.power = root.find('power').text.strip()
    self.initialBoron = root.find('initialBoron').text.strip()
    self.coretype = root.find('coretype').text.strip()
    self.XSdir = root.find('XSdir').text.strip()
    self.Depdir = root.find('Depdir').text.strip()
    self.DepHistory = root.find('DepHistory').text.strip()
    self.NFA = root.find('NFA').text.strip()
    self.NAxial = root.find('NAxial').text.strip()
    self.geometry = root.find('Geometry').text.strip()
    self.FA_Pitch = root.find('FA_Pitch').text.strip()
    self.FA_Power = root.find('FA_Power').text.strip()
    self.grid_x = root.find('grid_x').text.strip()
    self.grid_y = root.find('grid_y').text.strip()
    self.grid_z = root.find('grid_z').text.strip()
    self.neutmesh_x = root.find('neutmesh_x').text.strip()
    self.neutmesh_y = root.find('neutmesh_y').text.strip()
    self.BC = root.find('BC').text.strip()
    self.FAdict = []
    for FA in root.iter('FA'):
      self.FAdict.append(FA.attrib)
    self.XSdict =[]
    for xs in root.iter('XS'):
      self.XSdict.append(xs.attrib)

class PerturbedPaser():
  """
  Parse value in the perturbed xml file replaces the nominal values by the perturbed values.
  """
  def __init__(self, inputFile, workingDir, inputName, perturbDict):
    """
    Constructor.
      @ In, inputFiles, string, xml PARCS varibles that will be perturbed file
      @ In, workingDir, string, absolute path to the working directory
      @ In, inputName, string, inputname for PARCS input file
      @ In, perturbDict, dictionary, dictionary of perturbed variables
      @ Out, None
    """
    self.inputFile = inputFile #original perturb input file
    self.perturbDict = perturbDict #get from RAVEN
    self.workingDir = workingDir
    self.inputName = inputName
    # get perturbed value and create new xml file
    self.replaceOldFile()
  def replaceOldFile(self):
    """
    Replace orignal xml file with perturbed variables
    @ In, perturbed variables dictionary
    @ workingDir, absolute path to the working directory
    @ Out, None
    """
    perturbedID = []
    perturbedVal = []

    for key, value in self.perturbDict.items():
      id_ = ''.join([n for n in key if n.isdigit()])
      try:
        id_ = int(id_)
      except:
        ValueError('There is no id indication in variable from RAVEN, please check!')

      perturbedID.append(int(id_))
      perturbedVal.append(int(value))
    #sorting
    perturbedVal_sorted =  [val for _,val in sorted(zip(perturbedID,perturbedVal))]
    perturbedVal =  perturbedVal_sorted
    perturbedID = sorted(perturbedID)
    fullFile = os.path.join(self.inputFile)
    dorm = ET.parse(fullFile)
    root = dorm.getroot()
    self.data = root
    for child in self.data:
      idx = perturbedID.index(int(child.attrib['id']))
      child.attrib['FAid'] = str(perturbedVal[idx])
    writedata = ET.tostring(root,encoding='utf8', method='xml')
    newfile = open(self.inputFile, "w")
    newfile.write(writedata.decode())
    newfile.close()
  def generatePARCSInput(self, parameter):
    """
    Generate new input for PARCS to run
    @ In, parameter data in DataParser class
    @ Out, None
    """
    file_ = open(f"{self.workingDir}/{self.inputName}",'w')
    ## write initial lines
    file_.write(f"!****************************************************************************** \n")
    file_.write(f"CASEID {self.inputName}              OECD NEA MSLB  \n")
    file_.write(f"!****************************************************************************** \n")
    file_.write(f"CNTL \n")
    file_.write(f"      TH_FDBK   {parameter.THFlag}  \n")
    file_.write(f"      CORE_POWER {parameter.power}  \n")
    file_.write(f"      CORE_TYPE  {parameter.coretype}  \n")
    file_.write(f"      PPM        {parameter.initialBoron}  \n")
    file_.write(f"      DEPLETION  T  1.0E-3 T \n")
    file_.write(f"      TREE_XS    T  16 T  T  F  F  T  F  T  F  T  F  F  T  T  F  \n")
    file_.write(f"      BANK_POS   100 100 100 100 100 100  0 46.2 0 \n")
    file_.write(f"      XE_SM      1 1 \n")
    file_.write(f"      SEARCH     ppm \n")
    file_.write(f"      XS_EXTRAP  1.0 0.3 0.8 0.2 \n")
    file_.write(f"      PIN_POWER  T   \n")
    file_.write(f"      PRINT_OPT T F T T F T F F F T  F  F  F  F  F   \n")
    file_.write(f"PARAM \n")
    file_.write(f"      LSOLVER  1 1 20 \n")
    file_.write(f"      NODAL_KERN     NEMMG    ! FMFD    ! FDM \n")
    file_.write(f"      CMFD     2 \n")
    file_.write(f"      DECUSP   2 \n")
    file_.write(f"      INIT_GUESS 0 \n")
    file_.write(f"      conv_ss   1.e-6 5.e-5 1.e-3 0.001 !epseig,epsl2,epslinf,epstf \n")
    file_.write(f"      eps_erf   0.010 \n")
    file_.write(f"      eps_anm   0.000001 \n")
    file_.write(f"      nlupd_ss  5 5 1 \n")
    file_.write(f"GEOM \n")
    file_.write(f"      geo_dim {parameter.NFA} {parameter.NFA} {parameter.NAxial} 1 1 \n") # full core geomerty
    file_.write(f"      Rad_Conf                        !! \n")
    loadingPattern = getcoremap(parameter,[int(child.attrib['FAid']) for child in self.data], parameter.geometry)
    file_.write(loadingPattern)
    file_.write("\n")
    file_.write(f"      grid_x      {parameter.grid_x} \n")
    file_.write(f"      neutmesh_x  {parameter.neutmesh_x} \n")
    file_.write(f"      grid_y      {parameter.grid_y} \n")
    file_.write(f"      neutmesh_y  {parameter.neutmesh_y} \n")
    file_.write(f"      grid_z      {parameter.grid_z} \n")
    file_.write(f"      Boun_cond   {parameter.BC} \n")
    tmp= []
    for fa in parameter.FAdict:
      if fa['name'].lower() != 'none' or float(fa['FAid'])>=0:
        file_.write(f"      assy_type   {fa['type']}   {fa['structure']} \n")
    file_.write("\n")
    ##create pin calculation map
    pinmap = loadingPattern
    for fa in parameter.FAdict:
      if fa['name'].lower() == 'none' :
        pinmap = pinmap.replace(fa['type'],"  ")
      elif fa['name'].lower() == 'ref' :
        pinmap = pinmap.replace(fa['type']," 0")
    file_.write("\n")
    file_.write(f"     pincal_loc \n")
    file_.write(f"                        \n")
    file_.write(pinmap)
    file_.write("\n")
    file_.write("\n")
    file_.write(f"TH \n")
    file_.write(f"      unif_th         0.7  600.0  300.0 \n")
    file_.write(f"FDBK \n")
    file_.write(f"      fa_powpit       {parameter.FA_Power}   {parameter.FA_Pitch} \n")
    file_.write("\n")
    file_.write(f"DEPL \n")
    file_.write(f"      TIME_STP  {parameter.DepHistory}  \n")
    file_.write(f"      INP_HST   '../../{parameter.Depdir}/boc_exp_fc.dep' -2 1 \n")
    for xs in parameter.XSdict:
      file_.write(f"      PMAXS_F   {xs['id']} '../../{parameter.XSdir}/{xs['name']}'                 {xs['id']}   \n")
    file_.write(f".  \n")
    file_.close()
# Outside functions
def findType(FAid,FAdict):
  """
  Get type of FA ID
  """
  FAtype = [id['type'] for id in FAdict if id['FAid']==str(FAid)][0]
  return FAtype
def getcoremap(parameter, FAID, geometrykey):
  """
  Genrate Loading Pattern
  @IN: DataParser class
  @IN: FAID sorted list, geometry key for full or quater core
  @OUT: Loading Pattern
  """
  FAdict = parameter.FAdict
  maxType = max([id['type'] for id in FAdict])
  numberSpaces = len(str(maxType)) + 2
  emptyMap=[]
  if geometrykey.lower()=='full':
    rows, cols=17,17
    x_start, y_start = 9, 9
  elif geometrykey.lower() =='quater':
    rows, cols=9,9
    x_start, y_start = 1, 1
  else:
    raise ValueError("No available geometry key. Only full or quater core is supported")
  for i in range(rows):
      col = []
      for j in range(cols):
          col.append(0)
      emptyMap.append(col)
  idx_ = 0
  val = FAID[idx_]
  val = findType(val,FAdict)
  for x in range(x_start,x_start+9): # 17x17 core
    for y in range (y_start,x+1):
      val = FAID[idx_]
      val = findType(val,FAdict)
      if geometrykey.lower()=='full':
        outCoordianate = get_index_full(x,y,x_start,y_start)
      else:
        outCoordianate = get_index_quater(x,y)
      for i in outCoordianate:
        emptyMap[i[0]-1][i[1]-1]=val
      idx_ = idx_+1
  loadingPattern = "      "
  for i in emptyMap:
      for j in i:
          value=j
          str_ = f"{value}"
          loadingPattern += f"{str_.rjust(numberSpaces)}"
      loadingPattern += "\n"
      loadingPattern += "      "
  return loadingPattern
def get_index_full(x,y, x0, y0):
  """
  Get the index of symetric element in a 1/8 th symmetric core map
  to a full core map
  Input: x,y coordinate of the element
        x0,y0 coordinate of the center element
  Output: list of indices [(x,y)]
  """
  deltaX = x-x0
  deltaY = y-y0
  temparray = []
  temparray.append([x0+deltaX,y0+deltaY])
  temparray.append([x0+deltaX,y0-deltaY])
  temparray.append([x0-deltaX,y0+deltaY])
  temparray.append([x0-deltaX,y0-deltaY])
  temparray.append([x0+deltaY,y0+deltaX])
  temparray.append([x0+deltaY,y0-deltaX])
  temparray.append([x0-deltaY,y0+deltaX])
  temparray.append([x0-deltaY,y0-deltaX])

  #remove duplicate
  outarray = []
  [outarray.append(i) for i in temparray if i not in outarray]
  return outarray
def get_index_quater(x,y):
    """
    Get the index of symetric element in a 1/8 th symmetric core map
    to quater core
    Input: x,y coordinate of the element
    Output: list of indices [(x,y)]
    """
    temparray = []
    temparray.append([x,y])
    temparray.append([y,x])

    #remove duplicate
    outarray = []

    [outarray.append(i) for i in temparray if i not in outarray]
    return outarray