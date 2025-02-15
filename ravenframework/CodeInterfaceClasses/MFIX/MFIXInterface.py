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
Created on February 14, 2025

@author: wangc

comments: Interface for MFIX Simulation
"""
import os
import copy

import numpy as np
# import matplotlib.pyplot as plt
import pyvista as pv
import glob
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
# import time
# from sklearn import preprocessing as pp

from ..Generic.GenericCodeInterface import GenericCode
from .MFIXParser import MFIXParser

class MFIX(GenericCode):
  """MFIX RAVEN interface
  """

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()


    self.inputExtensions = ['mfx']
    self.outputExtensions = ['vtp', 'vtu']
    self.fixedOutFileName = None
    self.caseName = None

    # Known, changable variables
    self.deltaH = 0.001  # set value for the change in height
    self.deltaVol = 0.1  # volume fraction assumed to be the upper limit for the bed
    self.slopeStop = self.deltaVol / self.deltaH  # calculate the limit on the change in slope to determine bed height
    self.dPart = 2.5e-4  # particle volume (m)
    # particle diameter: d_p0(1) = 6.9e-4
    # self.rTol = self.dPart / 1000  # tolerance for particle diameter
    self.rTol = 2.e-3 # TODO read from MFIX input file d_p0(1) * 2
    self.rHem = 0.0045  # radius of hemispherical section at the bottom of the cone
    self.hOff = self.rHem / np.tan(np.pi / 6)  # height of the cone at the bottom of the cone ( height of the hemispherical section)
    self.nYMesh = 115
    self.basePartFile = 'BACKGROUND_IC_*.vtp' # Polygonal data, 2D grid, like the unstructured grid, but there are no polyhedra, but only flat polygons.
    self.cellPartFile = 'X_SLICE_*.vtu' # Unstructured grid: 2D or 3D grid; for every grid point all three coordinates and for each grid cell all constituent points and the cell shape are given
    self.moveAgeWindow = 5
    self.errTol = 0.0001
    # self.heightBin = 0.01
    self.numParticle = 3
    self.heightBin = self.numParticle * self.dPart
    self.coneRadius = 0.0254 #meters
    self.coneHeight = 0.045 #meters
    self.slantHeight = 0.0507204 # meters
    self.cylinderAboveConeHeight = 0.17780 + self.coneHeight
    self.translationVector = np.asarray([0, 0.022, 17.728])
    self._bins = None
    self._dataSet = xr.Dataset()

  def addInputExtension(self,exts):
    """
      This method adds a list of extension the code interface accepts for the input files
      @ In, exts, list, list or other array containing accepted input extension (e.g.[".i",".inp"])
      @ Out, None
    """
    for e in exts:
      if e not in self.ignoreInputExtensions:
        self.inputExtensions.append(e)

  def addDefaultExtension(self):
    """
      The Generic code interface does not accept any default input types.
      @ In, None
      @ Out, None
    """
    self.addInputExtension(['mfx'])

  def _readMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this class and initialize some members
      based on inputs
      @ In, xmlNode, xml.etree.ElementTree.Element, xml element node
      @ Out, None
    """
    GenericCode._readMoreXML(self, xmlNode)

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
    returnCommand = super().generateCommand(inputFiles,executable,clargs=clargs, fargs=fargs, preExec=preExec)
    return returnCommand

  def finalizeCodeOutput(self, command, output, workingDir):
    """
      This method is called by the RAVEN code at the end of each run (if the method is present).
      It can be used for those codes, that do not create CSV files to convert the whatever output format into a csv
      @ In, command, string, the command used to run the just ended job (in general, not used, it is the first argument returned by generateCommand)
      @ In, output, string, the Output name root without the file extension
      @ In, workingDir, string, current working dir
      @ Out, output, string or dict, optional, if present and string:
                                                  in case the root of the output file gets changed in this method (and a CSV is produced);
                                                if present and dict:
                                                  in case the output of the code is directly stored in a dictionary and can be directly used
                                                  without the need that RAVEN reads an additional CSV
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
                    # 'avg_part_bed_vx':[],
                    # 'avg_part_bed_vy':[],
                    # 'avg_part_bed_vz':[],
                    # 'avg_part_spout_vx':[],
                    # 'avg_part_spout_vy':[],
                    # 'avg_part_spout_vz':[],
                    # 'bed_height':[],
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
        partBase = pv.read(filename)  # reading the particle file
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

      #########################################################
      # Added by Congjian for explore average void fractions
      avgVolFracDict[timeVar] = np.atleast_1d(avgVolFrac[:, 3])
      cowDict[timeVar] = np.atleast_1d(cow)
      #########################################################

      sizeVolFrac = len(volFrac)  # finding the number of volume fractions
      sizeCoord = len(centCoord[:, 0])  # finding the number of cell center coordinate locations

      if sizeVolFrac == sizeCoord:  # check to make sure the number of volume fractions and cell centers is the same

        # Old way to compute bed edge particles
        ####################################################################
        # bedPartMask = part[:, 1] <= hBed
        # spoutPartMask = ~bedPartMask
        # bedPart = np.hstack((part[bedPartMask], partVel[bedPartMask]))
        # spoutPart = np.hstack((part[spoutPartMask], partVel[spoutPartMask]))
        # if len(bedPart) > 0:
        #   xBar, yBar, zBar = np.mean(bedPart[:, :3], axis=0)
        #   bedPart[:, :3] -= [xBar, yBar, zBar]
        #   hBed -= yBar
        #   spoutPart[:, :3] -= [xBar, yBar, zBar]
        #   yMin = np.min(bedPart[:, 1])
        #   bedPart[:, 1] -= yMin
        #   spoutPart[:, 1] -= yMin
        #   hBed -= yMin
        #   maskEdge = (bedPart[:, 0]**2 + bedPart[:, 2]**2)**0.5 >= np.tan(np.pi / 6) * (bedPart[:, 1] + self.hOff) - self.dPart - self.rTol
        #   bedEdge = bedPart[maskEdge]
        #   heights[i, 0] = timeVar
        #   heights[i, 1] = hBed
        #   bedEdgeDict[timeVar] = bedEdge
        ##################################################################################

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
        # outputResults['bed_height'].append(hBed)
        # outputResults['avg_part_bed_vx'].append(np.average(bedPart[:,3]))
        # outputResults['avg_part_bed_vy'].append(np.average(bedPart[:,4]))
        # outputResults['avg_part_bed_vz'].append(np.average(bedPart[:,5]))
        outputResults['avg_part_bedEdge_vx'].append(np.average(bedEdgePartVel[:,0]))
        outputResults['avg_part_bedEdge_vy'].append(np.average(bedEdgePartVel[:,1]))
        outputResults['avg_part_bedEdge_vz'].append(np.average(bedEdgePartVel[:,2]))
        # outputResults['avg_part_spout_vx'].append(np.average(spoutPart[:,3]))
        # outputResults['avg_part_spout_vy'].append(np.average(spoutPart[:,4]))
        # outputResults['avg_part_spout_vz'].append(np.average(spoutPart[:,5]))

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

    voidFracData = xr.DataArray(np.asarray(voidFracData).T, coords=[self._bins, outputResults['time']], dims=['height', 'time'])
    bedEdgeVxData = xr.DataArray(np.asarray(bedEdgeVxData).T, coords=[self._bins, outputResults['time']], dims=['height', 'time'])
    bedEdgeVyData = xr.DataArray(np.asarray(bedEdgeVyData).T, coords=[self._bins, outputResults['time']], dims=['height', 'time'])
    bedEdgeVzData = xr.DataArray(np.asarray(bedEdgeVzData).T, coords=[self._bins, outputResults['time']], dims=['height', 'time'])

    self._dataSet['void_frac'] = voidFracData
    self._dataSet['bed_edge_vx'] = bedEdgeVxData
    self._dataSet['bed_edge_vy'] = bedEdgeVyData
    self._dataSet['bed_edge_vz'] = bedEdgeVzData

    # hBedAvg = np.average(heights[:,1])
    # _, allEdgeSpaceAverage, avgAllEdgeSpaceAverage = self.calculateEdgeVelocityProfile(bedEdgeDict, hBedAvg)

    # convert list to numpy array
    for key, val in outputResults.items():
      outputResults[key] = np.asarray(val)

    # # TODO: update the calculation 'avg_part_bedEdge_vy'
    # outputResults['avg_edge_velocity'] = allEdgeSpaceAverage[:, 1]

    df = pd.DataFrame(outputResults)
    df.to_csv('out_with_edge.csv')

    # df_avgVolFrac = pd.DataFrame(avgVolFracDict, index=bins)
    # df_cow = pd.DataFrame(cowDict, index=bins[0:len(cow)])
    # df_avgVolFrac.to_csv('average_void_fraction.csv')
    # df_cow.to_csv('moving_average_void_fraction.csv')
    # # df_avgVolFrac.plot(legend=False, style=['-']*len(avgVolFracDict))
    # df_avgVolFrac.iloc[:,200:].plot(legend=False, style=['-']*len(avgVolFracDict))
    # plt.show()
    # df_cow.iloc[:,100:].plot(legend=False, style=['-']*len(avgVolFracDict))
    # plt.show()

    # Compute Edge Velocity Profile (in original code)
    # First compute the bed average heights (average over all perturbations and time)
    # Then call calculateEdgeVelocityProfile to compute the edge velocity profile
    # return is (numSteps*numBins, 3) with columns: BedHeight, Average y-velocity, TimeVar

    # Suggested way to compute:
    # compute the bed average heights for each run
    # Then calculateEdgeVelocityProfile and compute the average edge velocity profile over bins?
    # return is (numSteps, 3) with columns: BedHeight, Average y-velocity, TimeVar

    return outputResults

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
