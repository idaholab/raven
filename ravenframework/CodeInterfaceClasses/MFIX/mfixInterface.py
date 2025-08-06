# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Created on April 9, 2025

@author: wangc
"""
import numpy as np
import pyvista as pv
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import xarray as xr
from ravenframework.CodeInterfaceClasses.Generic.GenericCodeInterface import GenericCode
import pickle

class MFIX(GenericCode):
  """
    MFIX RAVEN interface
  """

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    GenericCode.__init__(self)
    self.inputExtensions = ['mfx']
    self.outputExtensions = ['vtp', 'vtu']
    self.fixedOutFileName = None
    self.caseName = None
    self._bins = None
    self._dataSet = xr.Dataset()

  def _readMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this class and initialize some members
      based on inputs
      @ In, xmlNode, xml.etree.ElementTree.Element, xml element node
      @ Out, None
    """
    GenericCode._readMoreXML(self, xmlNode)
    for child in xmlNode:
      if child.tag == 'deltaH':
        if child.text != None:           self.deltaH = float(child.text)
        else: pass
      elif child.tag == 'deltaVol':
        if child.text != None:           self.deltaVol =  float(child.text)
        else: pass
      elif child.tag == 'partVol':
        if child.text != None:           self.dPart =  float(child.text)
        else: pass
      elif child.tag == 'radHemCon':
        if child.text != None:           self.rHem =  float(child.text)
        else: pass
      elif child.tag == 'nYMesh':
        if child.text != None:           self.nYMesh =  int(child.text)
        else: pass
      elif child.tag == 'basePartFile':
        if child.text != None:           self.basePartFile =  child.text
        else: pass
      elif child.tag == 'cellPartFile':
        if child.text != None:           self.cellPartFile =  child.text
        else: pass
      elif child.tag == 'moveAgeWindow':
        if child.text != None:           self.moveAgeWindow =  int(child.text)
        else: pass
      elif child.tag == 'errTol':
        if child.text != None:           self.errTol =  float(child.text)
        else: pass
      elif child.tag == 'numParticle':
        if child.text != None:           self.numParticle =  int(child.text)
        else: pass
      elif child.tag == 'coneRadius':
        if child.text != None:           self.coneRadius =  float(child.text)
        else: pass
      elif child.tag == 'coneHeight':
        if child.text != None:           self.coneHeight =  float(child.text)
        else: pass
      elif child.tag == 'slantHeight':
        if child.text != None:           self.slantHeight =  float(child.text)
        else: pass
      elif child.tag == 'translationVector':
        if child.text != None:           self.translationVector =  np.array([float(i) for i in child.text.split(',')])
        else: pass

    self.slopeStop = self.deltaVol / self.deltaH
    self.hOff = self.rHem / np.tan(np.pi / 6)
    self.heightBin = self.numParticle * self.dPart
    self.cylinderAboveConeHeight = 0.17780 + self.coneHeight

  def initialize(self, runInfo, oriInputFiles):
    """
      Method to initialize the run of a new step
      @ In, runInfo, dict,  dictionary of the info in the <RunInfo> XML block
      @ In, oriInputFiles, list, list of the original input files
      @ Out, None
    """
    super().initialize(runInfo, oriInputFiles)
    with open(oriInputFiles[0].getAbsFile()) as mfix_inputFile: #NOTE: Rather than calling oriInputFiles[0], can we call MFiX input file?
      lines = mfix_inputFile.readlines()
      for line in lines:
        if 'nodesi' in line:
          try:
            words = line.split()
            nodesi = int(words[-1])
          except ValueError:
            raise IOError('The nodesi entry in the MFiX input file appears to be missing or not a numeric value. Please verify your MFiX input file.' )
        if 'nodesj' in line:
          try:
            words = line.split()
            nodesj = int(words[-1])
          except ValueError:
            raise IOError('The nodesj entry in the MFiX input file appears to be missing or not a numeric value. Please verify your MFiX input file.' )
        if 'nodesk' in line:
          try:
            words = line.split()
            nodesk = int(words[-1])
          except ValueError:
            raise IOError('The nodesk entry in the MFiX input file appears to be missing or not a numeric value. Please verify your MFiX input file.' )

    if runInfo['NumThreads'] == nodesi*nodesj*nodesk:
      pass
    else:
      raise IOError('\n''The number of thread in runInfo node of RAVEN input (i.e., <NumThreads>) MUST be identical with the multiplication of nodesi, nodesj, and nodesk in the MFiX input file. Please either verify your MFiX input file or adjust the number in <NumThreads> of <RunInfo>.' )

  def generateCommand(self,inputFiles,executable,clargs=None, fargs=None, preExec=None):
    """
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the auxiliary input file variables the user can specify in the input (e.g. under the node < Code >< fileargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ In, preExec, string, optional, a string the command that needs to be pre-executed before the actual command here defined
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    if clargs==None:
      raise IOError('No input file was specified in clargs!')
    extsClargs = list(ext[0][0] for ext in clargs['input'].values() if len(ext) != 0)
    extsFargs  = list(ext[0] for ext in fargs['input'].values())
    usedExts = extsClargs + extsFargs
    if len(usedExts) != len(set(usedExts)):
      raise IOError('GenericCodeInterface cannot handle multiple input files with the same extension.  You may need to write your own interface.')
    for inf in inputFiles:
      ext = '.' + inf.getExt() if inf.getExt() is not None else ''
      try:
        usedExts.remove(ext)
      except ValueError:
        pass
    if len(usedExts) != 0:
      raise IOError('Input extension',','.join(usedExts),'listed in XML node Code, but not found in the list of Input of <Files>')

    self.rTol = 2.e-3                               # TODO read from MFIX input file d_p0(1) * 2

    def getFileWithExtension(fileList,ext):
      """
      Just a script to get the file with extension ext from the fileList.
      @ In, fileList, the string list of filenames to pick from.
      @ Out, ext, the string extension that the desired filename ends with.
      """
      found = False
      for index,inputFile in enumerate(fileList):
        if inputFile.getExt() == ext:
          found=True
          break
      if not found:
        raise IOError('No InputFile with extension '+ext+' found!')
      return index,inputFile

    #prepend
    todo = ''
    todo += clargs['pre']+' '
    todo += executable
    # todo += inputFiles[0]
    index=None
    #inputs
    for flag,elems in clargs['input'].items():
      if flag == 'noarg':
        for elem in elems:
          ext, delimiter = elem[0], elem[1]
          idx,fname = getFileWithExtension(inputFiles,ext.strip('.'))
          todo += delimiter + fname.getAbsFile()
          if index == None:
            index = idx
        continue
      todo += ' '+flag
      for elem in elems:
        ext, delimiter = elem[0], elem[1]
        idx,fname = getFileWithExtension(inputFiles,ext.strip('.'))
        todo += delimiter + fname.getFilename()
        if index == None:
          index = idx
    #outputs
    #FIXME I think if you give multiple output flags this could result in overwriting
    self.caseName = inputFiles[index].getBase()
    outFile = 'out~'+self.caseName
    if 'output' in clargs:
      todo+=' '+clargs['output']+' '+outFile
    if self.fixedOutFileName is not None:
      outFile = self.fixedOutFileName
    todo+=' '+clargs['text']
    #postpend
    todo+=' '+clargs['post']
    returnCommand = [('parallel',todo)],outFile
    print('Execution Command: '+str(returnCommand[0]))
    return returnCommand

  def finalizeCodeOutput(self, command, output, workingDir):
    """
      This method is called by the RAVEN code at the end of each run (if the method is present).
      It can be used for those codes, that do not create CSV files to convert the whatever output format into a csv
      @ In, command, string, the command used to run the just ended job (in general, not used, it is the first argument returned by generateCommand)
      @ In, output, string, the Output name root without the file extension
      @ In, workingDir, string, current working dir
      @ Out, output, string or dict, optional,
      if present and string:
        in case the root of the output file gets changed in this method (and a CSV is produced);
      if present and dict:
        in case the output of the code is directly stored in a dictionary and can be directly used without the need that RAVEN reads an additional CSV
    """
    basePartFile = os.path.join(workingDir, self.basePartFile)
    partFiles = glob.glob(basePartFile)
    sizeFiles = np.size(partFiles)
    heights = np.zeros([sizeFiles, 2]) # matrix for population with the heights of the bed against time: col 0: normalized time index (time step num/100); col 1: bed heights
    bedEdgeDict = {}

    avgVolFracDict = {}
    cowDict = {}

    voidFracData = []
    bedEdgeVxData = []
    bedEdgeVyData = []
    bedEdgeVzData = []

    outputResults = {'time':[],
                    'avg_part_bedEdge_vx':[],
                    'avg_part_bedEdge_vy':[],
                    'avg_part_bedEdge_vz':[]
                    }

    for i, filename in enumerate(sorted(partFiles)):
      num = filename.split('_')[-1].split('.')[0]
      timeVar = int(num) / 100
      cellFile = self.cellPartFile.replace('*', num)
      cellFile = os.path.join(workingDir, cellFile)
      try:
        mesh = pv.read(cellFile)  # reading the cell file
        # mesh.plot()
      except:
        print('Skipping %s because %s does not exist' %(filename, cellFile))

      try:
        partBase = pv.read(filename)  # reading the particle files
      except:
        print('Skipping %s because %s does not exist' %('The Code', filename))

      part = partBase.points  # extracting the center locations for the particles

      # Available variables in current model: ['Velocity Magnitude', 'Diameter', 'Velocity']
      if 'Velocity' not in partBase.array_names:
        raise IOError(f"Variable 'Velocity' is not present in file {filename}")
      partVel = partBase.get_array("Velocity")  # extracting the velocity for the particles
      centCoord = mesh.cell_centers().points  # extracting the cell center locations
      # Available variables in current model: ['EP_G', 'U_G', 'V_G', 'W_G']
      # EP_G: void fraction; U_G, V_G, and W_G: Gas velocity
      if 'EP_G' not in mesh.array_names:
        raise IOError(f"Variable 'EP_G' is not present in file {cellFile}")
      volFrac = mesh.get_array('EP_G')  # extracting the cell volume fractions

      yMinCell = np.min(centCoord[:, 1])  # finding the minimum y value for cells
      yMaxCell = np.max(centCoord[:, 1])  # finding the maximum y value for cells
      bins = np.linspace(yMinCell, yMaxCell, num=self.nYMesh, endpoint=True)  # creating a linspace with the same number of cells as the simulation

      # bins = np.linspace(yMinCell, yMaxCell, num=self.nYMesh)  # creating a linspace with the same number of cells as the simulation
      self._bins = bins
      avgVolFrac = self.processVolumeFraction(centCoord, volFrac, bins)

      # utilizing moving average to compute the average void fraction, and use it to determine the bed height
      cow = self.movingAvg(avgVolFrac[:, 3], self.moveAgeWindow)
      err, EPGBed, hBed = self.processError(avgVolFrac, cow)

      sizeVolFrac = len(volFrac)        # finding the number of volume fractions
      sizeCoord = len(centCoord[:, 0])  # finding the number of cell center coordinate locations

      if sizeVolFrac == sizeCoord:      # check to make sure the number of volume fractions and cell centers is the same

        # move the bottom of the cone to (0, 0, 0)
        normPart = part + self.translationVector
        x = normPart[:, 0]
        z = normPart[:, 1]
        y = normPart[:, 2]

        # distance from the z-axis
        r = np.sqrt(x**2 + y**2)

        withinCone = (z>=0) & (z <= self.coneHeight)
        coneR = self.coneRadius * (z/self.coneHeight)
        nearConeSurface = withinCone & (np.abs(r-coneR) <= self.rTol)

        withinCylinder = (z > self.coneHeight) & (z <= self.cylinderAboveConeHeight)
        nearCylinderSurface = withinCylinder & (np.abs(r-self.coneRadius) <= self.rTol)

        maskEdge = nearConeSurface | nearCylinderSurface

        bedEdgePart = normPart[maskEdge]
        bedEdgePartVel = partVel[maskEdge]
        bedEdgeDict[timeVar] = bedEdgePartVel
        zMask = z[maskEdge]

        avgBedEdgePartVel = self.calculateEdgeVelocityWithGivenBins(zMask, bedEdgePartVel, bins)

        # collected data: bedEdge, bedPart, spoutPart for each file
        outputResults['time'].append(timeVar)
        outputResults['avg_part_bedEdge_vx'].append(np.average(bedEdgePartVel[:,0]))
        outputResults['avg_part_bedEdge_vy'].append(np.average(bedEdgePartVel[:,1]))
        outputResults['avg_part_bedEdge_vz'].append(np.average(bedEdgePartVel[:,2]))

        # Save the void fraction data
        for i, bin in enumerate(bins):
          voidFracName = 'avg_void_frac_at_bin_index_' + str(i)
          if voidFracName not in outputResults:
            outputResults[voidFracName] = []
          outputResults[voidFracName].append(avgVolFrac[i,3])
          for j, velName in enumerate(['vx', 'vy', 'vz']):
            avgBedEdgePartVelName = 'avg_bed_edge_' + velName + '_at_bin_index_' + str(i)
            if avgBedEdgePartVelName not in outputResults:
              outputResults[avgBedEdgePartVelName] = []
            outputResults[avgBedEdgePartVelName].append(avgBedEdgePartVel[i,j])

        voidFracData.append(avgVolFrac[:,3])
        bedEdgeVxData.append(avgBedEdgePartVel[:,0])
        bedEdgeVyData.append(avgBedEdgePartVel[:,1])
        bedEdgeVzData.append(avgBedEdgePartVel[:,2])

    # convert list to numpy array
    for key, val in outputResults.items():
      outputResults[key] = np.asarray(val)

    # # TODO: update the calculation 'avg_part_bedEdge_vy'
    # outputResults['avg_edge_velocity'] = allEdgeSpaceAverage[:, 1]

    df = pd.DataFrame(outputResults)
    # df.to_csv(os.path.join(workingDir,r'out_with_edge.csv'))
    df.to_csv(os.path.join(workingDir,r'out~MFIX_RAVEN_Temp.csv'))


    file_path = workingDir + '/dataset.pkl'
    # with open('dataset.pkl', 'wb') as f:
    with open(file_path, 'wb') as f:
      pickle.dump(self._dataSet, f, protocol=-1)

  def checkForOutputFailure(self, output, workingDir):
    """
      This method is called by RAVEN at the end of each run if the return code is == 0.
      This method needs to be implemented for the codes that, if the run fails, return a return code that is 0
      This can happen in those codes that record the failure of the job (e.g. not converged, etc.) as normal termination (returncode == 0)
      This method can be used, for example, to parse the output file looking for a special keyword that testifies that a particular job got failed
      (e.g. in RELAP5 would be the keyword "********")
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, failure, bool, True if the job is failed, False otherwise
    """
    failure = False
    return failure

  @staticmethod
  def movingAvg(x, w):
    """
    Computes the moving average of an array using NumPy.

    Args:
        x (np.ndarray): Input array of data.
        w (int): The window size for the moving average.

    Returns:
        np.ndarray: The array containing the moving averages.
    """
    return np.convolve(x, np.ones(w), 'valid') / w

  def processVolumeFraction(self, centCoord, volFrac, bins):
      """
      Computes the average volume fraction for each calculated bin.

      Args:
          centCoord (np.ndarray): Array of cell center coordinates.
          volFrac (np.ndarray): Array of volume fractions.
          bins (np.ndarray): Array of bin edges for the mesh.

      Returns:
          np.ndarray: Array containing the average volume fractions for each bin.
      """
      # Initialize the output array for storing average volume fractions and related data
      avgVolFrac = np.zeros((self.nYMesh, 4))
      avgVolFrac[:, 0] = bins
      digitized = np.digitize(centCoord[:, 1], bins)

      for i in range(1, self.nYMesh):
        mask = digitized == i
        avgVolFrac[i, 1] = np.sum(volFrac[mask])
        avgVolFrac[i, 2] = mask.sum()  # More efficient sum

      # Efficiently calculate average volume fractions
      nonZeroMask = avgVolFrac[:, 2] != 0
      avgVolFrac[nonZeroMask, 3] = avgVolFrac[nonZeroMask, 1] / avgVolFrac[nonZeroMask, 2]

      return avgVolFrac

  def calculateEdgeVelocityWithGivenBins(self, z, bedEdge, bins):
    """
    Calculate the edge velocity for bed particles for given time step.

    This function processes the edge particles of the bed, binning them by height,
    and calculating the average velocity and slope of the velocity profile across
    the bed height.

    Args:
        bedEdge (np.ndarray): The array of edge particle data with positions and velocities.

    Returns:
        np.ndarray: An array of the average y-velocity for all edge particles at different heights.
        NumOfRows: NumBinsInBedHeight * NumTimeSteps
    """
    nBed = len(bins)
    binIndices = np.searchsorted(bins, z, side='left')
    binIndices = np.clip(binIndices, 0, nBed - 1)
    binnedVelocities = np.zeros((nBed, 3))
    np.add.at(binnedVelocities, binIndices, bedEdge)
    binCounts = np.bincount(binIndices, minlength=nBed)

    nonzeroBins = binCounts > 0
    averageVelocities = np.zeros_like(binnedVelocities)
    averageVelocities[nonzeroBins] = binnedVelocities[nonzeroBins] / binCounts[nonzeroBins][:, None]

    return averageVelocities


  def calculateEdgeVelocity(self, bedEdge, hBedAvg):
    """
    Calculate the edge velocity for bed particles for given time step.

    This function processes the edge particles of the bed, binning them by height,
    and calculating the average velocity and slope of the velocity profile across
    the bed height.

    Args:
        bedEdge (np.ndarray): The array of edge particle data with positions and velocities.

    Returns:
        np.ndarray: An array of the average y-velocity for all edge particles at different heights.
        Columns: BedHeight, Average y-velocity, TimeVar (repeated for different bed heights), the first two variables will be stacked for different TimeVar
        NumOfRows: NumBinsInBedHeight * NumTimeSteps
    """
    # Calculate the average bed height from the heights array
    maxHeight = hBedAvg
    nBed = int(np.ceil(maxHeight / self.heightBin))
    heightBins = np.linspace(0, maxHeight, num=nBed, endpoint=True)
    validMask = (bedEdge[:, 1] >= 0) & (bedEdge[:, 1] < maxHeight)
    validParticles = bedEdge[validMask]
    binIndices = np.searchsorted(heightBins, validParticles[:, 1], side='left')
    binIndices = np.clip(binIndices, 0, nBed - 1)
    binnedVelocities = np.zeros((nBed, 3))
    np.add.at(binnedVelocities, binIndices, validParticles[:, 3:6])
    binCounts = np.bincount(binIndices, minlength=nBed)

    nonzeroBins = binCounts > 0
    averageVelocities = np.zeros_like(binnedVelocities)
    averageVelocities[nonzeroBins] = binnedVelocities[nonzeroBins] / binCounts[nonzeroBins][:, None]
    averageVelocities = averageVelocities[nonzeroBins]
    validHeightBins = heightBins[nonzeroBins]
    # try:
    #   slopesY = np.gradient(averageVelocities[nonzeroBins, 1], validHeightBins)

    return averageVelocities, validHeightBins

  def calculateEdgeVelocityProfile(self, bedEdgeDict, hBedAvg):
    """
    Calculate the edge velocity profile for bed particles.

    This function processes the edge particles of the bed, binning them by height,
    and calculating the average velocity and slope of the velocity profile across
    the bed height. It also tracks the edge particle velocities over time.

    Args:
        bedEdge (np.ndarray): The array of edge particle data with positions and velocities.
        heights (np.ndarray): The array containing the processed bed heights at each time step.
        timeVar (float): The current time variable for associating with the velocity data.

    Returns:
        np.ndarray: An array of the average y-velocity for all edge particles at different heights.
        Columns: BedHeight, Average y-velocity, TimeVar (repeated for different bed heights), the first two variables will be stacked for different TimeVar
        NumOfRows: NumBinsInBedHeight * NumTimeSteps
    """
    allEdgeSpaceSorted2ColumnsList = []
    allEdgeSpaceAverage = np.zeros([1,2])

    for timeVar, bedEdge in bedEdgeDict.items():
      # Calculate the average bed height from the heights array
      averageVelocities, validHeightBins = self.calculateEdgeVelocity(bedEdge, hBedAvg)
      shape = validHeightBins.shape
      # Create an array filled with timeVar, matching the shape of the other arrays
      timeVarArray = np.full(shape, timeVar)
      edgeSpaceSorted2Columns = np.stack((validHeightBins, averageVelocities[:, 1], timeVarArray), axis=1)
      allEdgeSpaceSorted2ColumnsList.append(edgeSpaceSorted2Columns)
      # y velocity
      allEdgeSpaceAverage = np.append(allEdgeSpaceAverage, [[timeVar, np.mean(averageVelocities[:,1])]], axis=0)

    if allEdgeSpaceSorted2ColumnsList:  # Check that the list is not empty
      allEdgeSpaceSorted2ColumnsArray = np.concatenate(allEdgeSpaceSorted2ColumnsList, axis=0)

      # calculate the average edge velocity over all bins and time
      allEdgeSpaceAverage = allEdgeSpaceAverage[allEdgeSpaceAverage[:,0].argsort()]
      maxLen = len(allEdgeSpaceAverage)
      avgAllEdgeSpaceAverage = np.average(allEdgeSpaceAverage[100:maxLen, -1])
    else:
      print("No edge space data to concatenate.")
      # You might want to handle this case appropriately, e.g., by creating an empty array
      allEdgeSpaceSorted2ColumnsArray = np.empty((0, 3))
      allEdgeSpaceAverage = np.empty((0, 2))
      avgAllEdgeSpaceAverage = 0

    allEdgeSpaceAverage = np.delete(allEdgeSpaceAverage, 0, 0)
    return allEdgeSpaceSorted2ColumnsArray, allEdgeSpaceAverage, avgAllEdgeSpaceAverage


  def processError(self, avgVolFrac, cow):
    """
    Computes the error from a given tolerance between the average volume
    fraction and the moo-ving average version. Although I do want to talk to stoyer about the
    usefulness of this function,
    this seems to be really old logic and it may be integral to the calculations but I cant be certain.

    Args:
        avgVolFrac (np.ndarray): Array of average volume fractions.
        cow (np.ndarray): Moo-ving average array.

    Returns:
        tuple: A tuple containing arrays of errors and detected bed height information.
    """
    # Compute the absolute error between average volume fractions and the reference curve
    err = np.abs(avgVolFrac[:len(cow), 3] - cow)
    EPGBed = hBed = None
    # Detect the bed height based on error tolerance
    for i in range(1, len(err)):
      if np.mean(err[i:i + 10]) < self.errTol and err[i - 1] > self.errTol:
        EPGBed = cow[i - 1]
        hBed = avgVolFrac[i - 1, 0]
        break
    # make sure the bed height does not exceed the maximum height in the bins
    hBed = min(hBed, np.max(avgVolFrac[:, 0])) if hBed is not None else np.max(avgVolFrac[:, 0])
    return err, EPGBed, hBed
