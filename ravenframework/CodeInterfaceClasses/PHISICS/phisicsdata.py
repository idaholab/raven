"""
Created on July 25th, 2017
@author: rouxpn
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import os
import re
import csv
import xml.etree.ElementTree as ET
import numpy as np
from decimal import Decimal
from collections import defaultdict


class phisicsdata():
  """
    This class parses the PHISICS output of interest. The output of interest are printed in the a csv file.
  """

  def __init__(self, phisicsDataDict):
    """
      Reads the PHISICS output.
      @ In, phisicsDataDict, dictionary, dictionary of variables passed by the interface
      @ Out, None
    """
    # boolean: True means PHISICS RELAP coupled, False means PHISICS standalone
    self.phisicsRelap = phisicsDataDict['phiRel']
    # boolean: True means spatial RR are printed, False means spatial RR are not printed
    self.printSpatialRR = phisicsDataDict['printSpatialRR']
    # boolean: True means spatial fluxes are printed, False means spatial fluxes are not printed
    self.printSpatialFlux = phisicsDataDict['printSpatialFlux']
    # string: working directory
    self.workingDir = phisicsDataDict['workingDir']
    # string: mrtau output file name appended with the MPI number
    self.mrtauOutputFileMPI = []
    # dict: dictionary containing all the variables relative to PHISICS output
    phisicsDict = {}
    # dict: dictionary containing all the variables relative to mrtau output, if MRTAU is ran in standalone mode
    mrtauDict = {}
    # list: markers delimiters used for parsing
    markerList = ['Fission matrices of', 'Scattering matrices of','Multigroup solver ended!']
    # data
    self.data = defaultdict(list)
    if not self.phisicsRelap:
      self.instantOutputFileMPI, self.mrtauOutputFileMPI = self.fileOutName(phisicsDataDict)
    elif self.phisicsRelap and phisicsDataDict['numberOfMPI'] > 1:
      self.instantOutputFileMPI, self.mrtauOutputFileMPI = self.fileOutName(phisicsDataDict)
    else:
      # instantOutputFile is populated but needs to be renamed
      self.instantOutputFileMPI, self.mrtauOutputFileMPI = self.fileOutName(phisicsDataDict)
      self.instantOutputFileMPI = [phisicsDataDict['relapOut'] + '.o']
    if not phisicsDataDict['mrtauStandAlone']:
      cpuTime = self.getCPUtime(phisicsDataDict['numberOfMPI'])
      self.materialsDict = self.locateMaterialInFile(phisicsDataDict['numberOfMPI'])
      self.getNumberOfGroups()
      self.getNumberOfRegions()
    else:
      cpuTime = self.getCPUtimeMrtau(phisicsDataDict['output'])

    if not self.phisicsRelap:
      instantOutput = [
          'Dpl_INSTANT_' + phisicsDataDict['jobTitle'] + '_flux_mat.csv',
          'scaled_xs.xml'
      ]
    else:
      instantOutput = [
          'PHISICS_RELAP5_' + phisicsDataDict['jobTitle'] + '_flux_sub.csv',
          'PHISICS_RELAP5_' + phisicsDataDict['jobTitle'] +
          '_power_dens_sub.csv', 'scaled_xs.xml'
      ]

    self.cleanUp(phisicsDataDict['jobTitle'],
                 phisicsDataDict['mrtauStandAlone'])
    if not phisicsDataDict['mrtauStandAlone']:  # MRTAU/INSTANT are coupled
      mrtauTimeSteps = self.getMrtauInstantTimeSteps()
      instantTimeSteps = self.getInstantTimeSteps(instantOutput[0])
      # only run the following method if XS are perturbed
      if 'XS' in phisicsDataDict['pertVariablesDict']:
        xsLabelList, xsList = self.getAbsoluteXS(instantOutput[1])
      else:
        xsLabelList = ['NoXS']
        xsList = [0.0000]
    else:  # MRTAU is standalone mode
      mrtauTimeSteps = self.getMrtauTimeSteps(phisicsDataDict['mrtauFileNameDict']['atoms_csv'])
      self.getMrtauIsotopeList(phisicsDataDict['mrtauFileNameDict']['atoms_csv'])

    data = None

    for timeStepIndex in range(len(mrtauTimeSteps)):
      if not phisicsDataDict['mrtauStandAlone']:
        keff, errorKeff = self.getKeff()
        reactionRateInfoMPI = self.getReactionRates(
            phisicsDataDict['numberOfMPI'])
        reactionRateInfo = self.summedDictValues(reactionRateInfoMPI)
        if self.printSpatialRR:
          fissionMatrixInfoMPI = self.getMatrix(markerList[0], markerList[1],
                                                phisicsDataDict['numberOfMPI'])
          fissionMatrixInfo = self.summedDictValues(fissionMatrixInfoMPI)
        else:
          fissionMatrixInfo = {}
        if self.printSpatialFlux:
          if not self.phisicsRelap:
            fluxLabelList, fluxList, matFluxLabelList, matFluxList = self.getFluxInfo(
                instantOutput[0])
          else:
            fluxLabelList, fluxList = self.getFluxInfoPhiRel(instantOutput[0])
            powerDensLabelList, powerDensList = self.getFluxInfoPhiRel(
                instantOutput[1])
        else:
          fluxLabelList = ['fluxLabel']
          fluxList = [0.]
          matFluxLabelList = ['matFluxLabelList']
          matFluxList = [0.]
          powerDensLabelList = ['powerDensLabelList']
          powerDensList = [0.]
        depLabelList, depList, timeStepList, matList = self.getDepInfo(
            timeStepIndex, mrtauTimeSteps, phisicsDataDict['numberOfMPI'])
        decayLabelList, decayList = self.getDecayHeat(
            timeStepIndex, mrtauTimeSteps, phisicsDataDict['decayHeatFlag'])
        buLabelList, buList = self.getBurnUp(timeStepIndex, mrtauTimeSteps)
        phisicsDict['keff'] = keff
        phisicsDict['errorKeff'] = errorKeff
        phisicsDict['reactionRateInfo'] = reactionRateInfo
        phisicsDict['fissionMatrixInfo'] = fissionMatrixInfo
        phisicsDict['workingDir'] = self.workingDir
        phisicsDict['fluxLabelList'] = fluxLabelList
        phisicsDict['fluxList'] = fluxList
        phisicsDict['depLabelList'] = depLabelList
        phisicsDict['depList'] = depList
        phisicsDict['timeStepList'] = timeStepList
        phisicsDict['decayLabelList'] = decayLabelList
        phisicsDict['decayList'] = decayList
        phisicsDict['timeStepIndex'] = timeStepIndex
        phisicsDict['mrtauTimeSteps'] = mrtauTimeSteps
        phisicsDict['xsLabelList'] = xsLabelList
        phisicsDict['xsList'] = xsList
        phisicsDict['cpuTime'] = cpuTime
        phisicsDict['buLabelList'] = buLabelList
        phisicsDict['buList'] = buList

        if self.phisicsRelap:
          phisicsDict['powerDensList'] = powerDensList
          phisicsDict['powerDensLabelList'] = powerDensLabelList
        else:
          phisicsDict['matFluxLabelList'] = matFluxLabelList
          phisicsDict['matFluxList'] = matFluxList

        # collect data
        h, snapshoot = self.phisicsTimeStepData(phisicsDict, timeStepIndex, mrtauTimeSteps)
        if data is None:
          headers = h
          data = np.zeros((len(h),len(mrtauTimeSteps)))
        data[:,timeStepIndex] = snapshoot

      if phisicsDataDict['mrtauStandAlone']:
        decayHeatMrtau = self.getDecayHeatMrtau(
            timeStepIndex, mrtauTimeSteps,
            phisicsDataDict['mrtauFileNameDict']['decay_heat'])
        depList = self.getDepInfoMrtau(
            timeStepIndex, mrtauTimeSteps,
            phisicsDataDict['mrtauFileNameDict']['atoms_csv'])
        mrtauDict['workingDir'] = self.workingDir
        mrtauDict['decayHeatMrtau'] = decayHeatMrtau
        mrtauDict['depList'] = depList
        mrtauDict['timeStepIndex'] = timeStepIndex
        mrtauDict['mrtauTimeSteps'] = mrtauTimeSteps
        # collect data
        h, snapshoot = self.mrtauTimeStepData(mrtauDict)
        if data is None:
          headers, data = h, np.zeros((len(h),len(mrtauTimeSteps)))
        data[:,timeStepIndex] = snapshoot
    #store the data
    self.data = {var:data[i,:] for i,var in enumerate(headers)}

  def summedDictValues(self, nestedDict):
    """
      Sums the values from the deepest nest of a dictionary.
      @ In, nestedDict, dictionary, nested dictionaries of integers or floats
      @ Out, summedDict, dictionary, dictionary of integer or float
    """
    summedDict = lambda: defaultdict(summedDict)
    mySummedDict = summedDict()
    if not self.printSpatialRR:
      for parameter, mpi in nestedDict.items():
        sumString = '%.15E' % Decimal(
            sum([float(elm) for elm in mpi.values()]))
        mySummedDict[parameter] = sumString
    else:
      for group, region in nestedDict.items():
        for region, parameter in region.items():
          for parameter, mpi in parameter.items():
            sumString = '%.15E' % Decimal(
                sum([float(elm) for elm in mpi.values()]))
            mySummedDict[group][region][parameter] = sumString
    return mySummedDict

  def fileOutName(self, phisicsDataDict):
    """
      Puts in a list the INSTANT output file names based on the number of MPI.
      The format is title.o-MPI
      @ In, phisicsDataDict, dictionary, dictionary of variables passed by the interface
      @ Out, instantOutputFileMPI, list, list of INSTANT output file names
      @ Out, mrtauOutputFileMPI, list, list of MRTAU output file names
    """
    instantMPI = []
    mrtauMPI = []
    for mpi in range(phisicsDataDict['numberOfMPI']):
      mrtauMPI.append(
          os.path.join(
              self.workingDir,
              phisicsDataDict['mrtauFileNameDict']['atoms_csv'].split('.')[0] +
              '-' + str(mpi) + '.csv'))  # MRTAU files (numbers.csv)
      if not self.phisicsRelap:  # INSTANT files, no coupling with RELAP
        instantMPI.append(phisicsDataDict['instantOutput'] + '-' + str(mpi))
      else:  # INSTANT files, coupling with RELAP
        if mpi > 0:
          instantMPI.append('INSTout-' + str(mpi))
    return instantMPI, mrtauMPI

  def cleanUp(self, jobTitle, bool):
    """
      Removes the file that RAVEN reads for postprocessing.
      @ In, jobTitle, string, job title name
      @ In, bool, bool, True if mrtau is stand alone, false otherwise
      @ Out, None
    """
    if not bool:
      csvOutput = os.path.join(self.workingDir, jobTitle + '.csv')
      #csvOutput = os.path.join(self.workingDir, jobTitle+'-'+self.perturbationNumber+'.csv')
      if os.path.isfile(csvOutput):
        os.remove(csvOutput)
      with open(csvOutput, 'wb'):
        pass
    else:
      mrtauOutput = os.path.join(self.workingDir, 'mrtau.csv')
      #mrtauOutput = os.path.join(self.workingDir, 'mrtau-'+self.perturbationNumber+'.csv') # if mrtau standalone
      if os.path.isfile(mrtauOutput):
        os.remove(mrtauOutput)  # if mrtau standalone
      with open(mrtauOutput, 'wb'):
        pass

  def getAbsoluteXS(self, xsOutput):
    """
      Parses the absolute cross section output generated by PHISICS.
      @ In, xsOutput, string, filename of the XML file containing the absolute perturbed XS
      @ Out, labelList, list of the XS labels
      @ Out, valueList, list of the XS
    """
    labelList = []
    valueList = []
    tree = ET.parse(os.path.join(self.workingDir, xsOutput))
    root = tree.getroot()
    for materialXML in root.iter('library'):
      for isotopeXML in materialXML.iter('isotope'):
        reactionList = [j.tag for j in isotopeXML]
        for react in reactionList:
          for groupXML in isotopeXML.iter(react):
            individualGroup = [
                x.strip() for x in groupXML.attrib.get('g').split(',')
            ]
            individualGroupValues = [
                y.strip() for y in groupXML.text.split(',')
            ]
            for position in range(len(individualGroup)):
              labelList.append(
                  materialXML.attrib.get('lib_name') + '|' + isotopeXML.attrib.
                  get('id') + '|' + react + '|' + individualGroup[position])
              valueList.append(individualGroupValues[position])
    return labelList, valueList

  def getNumberOfGroups(self):
    """
      Gives the number of energy groups used in the PHISICS simulations.
      @ In, None
      @ Out, None
    """
    self.Ngroups = 0
    with open(
        os.path.join(self.workingDir, self.instantOutputFileMPI[0]),
        'r') as outfile:
      for line in outfile:
        if re.search(r'Number of elements of group', line):
          self.Ngroups = int(
              line.split()[-3]
          )  # pick the last occurence (last group), which is also the number of groups
        if re.search(r'Number of element  types', line):
          break

  def getNumberOfRegions(self):
    """
      Gives the number of spatial regions used in the PHISICS simulations.
      @ In, None
      @ Out, None
    """
    flagStart = 0
    count = 0
    subdomainList = []
    with open(
        os.path.join(self.workingDir, self.instantOutputFileMPI[0]),
        'r') as outfile:
      for line in outfile:
        if re.search(r'Subdomain volumes', line):
          flagStart = 1
        if re.search(r'Balance report for the primal solution', line):
          flagStart = 2
        if flagStart == 1:
          stringIsNumber = self.isNumber(line.split())
          if stringIsNumber:
            count = count + 1
            subdomainList.append(line.split()[0])
        if flagStart == 2:
          break
    self.Nregions = count
    return

  def getMrtauInstantTimeSteps(self):
    """
      Gets the time steps in the coupled MRTAU/INSTANT output.
      @ In, None
      @ Out, timeSteps, list
    """
    count = 0
    timeSteps = []
    nts = 0
    with open(os.path.join(self.workingDir, self.mrtauOutputFileMPI[0]),
              'r') as outfile:
      for line in outfile:
        if re.search(r'Material', line):
          count = count + 1
        if count == 1:
          stringIsFloatNumber = self.isFloatNumber(line.split(','))
          if stringIsFloatNumber:
            ts = float(line.split(',')[0])
            if timeSteps and ts <= float(timeSteps[-1]):
              break
            timeSteps.append(line.split(',')[0])
        if count > 1:
          break
    # Removes the first time step, t=0, to make the number of time steps match with the number of time steps in INSTANT.
    timeSteps.pop(0)
    return timeSteps

  def getMrtauTimeSteps(self, atomsInp):
    """
      Gets the time steps in the MRTAU standalone output.
      @ In, atomsInp, string, path to file pointed by the node <atoms_csv> in the lib path file
      @ Out, timeSteps, list, list of time steps
    """
    timeSteps = []
    with open(os.path.join(self.workingDir, atomsInp), 'r') as outfile:
      for line in outfile:
        stringIsFloatNumber = self.isFloatNumber(line.split(','))
        if stringIsFloatNumber:
          timeSteps.append(line.split(',')[0])
    return timeSteps

  def getMrtauIsotopeList(self, atomsInp):
    """
      Gets the isotope in the MRTAU stand alone output.
      @ In, atomsInp, string, path to file pointed by the node <atoms_csv> in the lib path file
      @ Out, None
    """
    self.isotopeListMrtau = []
    with open(os.path.join(self.workingDir, atomsInp), 'r') as outfile:
      for line in outfile:
        if re.search(r'TIME', line):
          line = re.sub(r'TIME\s+\(days\)', r'', line)
          line = re.sub(r' ', r'', line)
          line = re.sub(r'\n', r'', line)
          self.isotopeListMrtau = list(filter(None, line.split(',')))
          break

  def getInstantTimeSteps(self, input):
    """
      Gets the time steps in the INSTANT output.
      @ In, input, string, input file name
      @ Out, timeSteps, list, list of time steps
    """
    count = 0
    timeSteps = []
    with open(os.path.join(self.workingDir, input), 'r') as outfile:
      for line in outfile:
        if re.search(r'PHISICS', line):
          timeSteps.append(line.split(',')[line.split(',').index('Time') + 1])
    return timeSteps

  def reduceDigits(self, mrtauTimeSteps):
    """
      Reduces the number of digits in the MRTAU time steps.
      @ In, mrtauTimeSteps, list
      @ Out, timeSteps, list, list of time steps
    """
    timeSteps = []
    for i in range(len(mrtauTimeSteps)):
      timeSteps.append('%.6E' % (float(mrtauTimeSteps[i])))
    return timeSteps

  def getKeff(self):
    """
      Gets the multiplication factor.
      @ In, None
      @ Out, keff, string, floating number under string format of keff
      @ Out, errorKeff, string, floating number under string format of errorKeff
    """
    self.units = None
    keff = []
    with open(
        os.path.join(self.workingDir, self.instantOutputFileMPI[0]),
        'r') as outfile:
      for line in outfile:
        if re.search(r'k-effective at the last', line):
          keff = [line.split()[-1]]
        if re.search(r'error for the eigenvalue', line):
          errorKeff = [line.split()[-1]]
    return keff, errorKeff

  def isNumber(self, line):
    """
      Checks if a string is an integer.
      @ In, line, list, list of strings
      @ Out, isNumber, bool, True if integer, False otherwise
    """
    if line != []:
      try:
        numFloat = float(line[0])
        numInt = int(numFloat)
        return numInt == numFloat
      except ValueError:  # the instance is a string, for example
        return False

  def isFloatNumber(self, line):
    """
      Checks if a string is a float number.
      @ In, line, list, list of strings
      @ Out, isFloatNumber, bool, True if float, False otherwise
    """
    if line != []:
      try:
        float(line[0])
        return True
      except ValueError:
        return False

  def getReactionRates(self, numberOfMPI):
    """
      Gets the reactions rates, power for each group in PHISICS
      @ In, numberOfMPI, interger, number of MPI user-defined
      @ Out, myReactDict, dictionary, dictionary containing the reaction rate info
    """
    flagStart = 0
    count = 0
    reactDict = lambda: defaultdict(reactDict)
    myReactDict = reactDict()
    for mpi in range(numberOfMPI):  # parse all the segmented files
      if not (
          mpi >= numberOfMPI - 1 and self.phisicsRelap and numberOfMPI > 1
      ):  # the case numberOfMpi > 1 is meant to enter the if loop if PHISICS/RELAP5 is ran in series
        with open(
            os.path.join(self.workingDir, self.instantOutputFileMPI[mpi]),
            'r') as outfile:
          for line in outfile:
            if re.search(r'averaged flux\s+power',
                         line):  # beginning of the reaction rate matrix
              if mpi == 0:  # no need to repeat the mapping and parameter listing for each mpi segmented files
                numberingRR = self.mapColumns(line, count)
                self.paramList = []
                self.paramList = numberingRR.keys()
              flagStart = 1
            if re.search(r'Fission matrices of all',
                         line):  # end of the reaction rate matrix
              flagStart = 2
            if flagStart == 1:  # the parsing starts here. It ends as soon as flagStart = 2
              if self.printSpatialRR:  # if the spatial reaction rates are printed
                if re.search(r'\w+\s+\w+', line):
                  line = re.sub(r'[\|]', ' ', line)
                  line = line.split()
                  stringIsNumber = self.isNumber(line)
                  if stringIsNumber:
                    for reactRate in numberingRR:
                      groupNum = line[0]
                      regionNum = line[1]
                      if reactRate != 'Group' and reactRate != 'Region':
                        myReactDict[groupNum][regionNum][reactRate][
                            mpi] = line[numberingRR.get(reactRate)]
              else:  # The spatial reaction rates are not printed
                if re.search(r'Total', line):
                  for reactRate in numberingRR:
                    if reactRate != 'Group' and reactRate != 'Region':
                      myReactDict[reactRate][mpi] = line.split()[
                          numberingRR.get(reactRate)]
    if myReactDict != {}:
      return myReactDict

  def getMatrix(self, startStringFlag, endStringFlag, numberOfMPI):
    """
      Gets the fission matrix or scattering matrix in the PHISICS output.
      @ In, startStringFlag, string, string marker.
      @ In, endStringFlag, string, string marker.
      @ In, numberOfMPI, interger, number of MPI user-defined
      @ Out, myMatrixDict, dictionary, nested dictionary that contains the fission matrix information.
                                    Template: {group number(from):{region number:{group(to):{mpi:Vslue}}}}
    """
    flagStart = 0
    count = 0
    test = {}
    matrixDict = lambda: defaultdict(matrixDict)
    myMatrixDict = matrixDict()
    for mpi in range(numberOfMPI):  # parse all the segmented files
      with open(
          os.path.join(self.workingDir, self.instantOutputFileMPI[mpi]),
          'r') as outfile:
        for line in outfile:
          if re.search(
              startStringFlag,
              line):  # begining of the matrix portion (scattering or fission)
            flagStart = 1
          if re.search(endStringFlag, line):  # end of the matrix portion
            flagStart = 2
          if flagStart == 1:  # starts parsing the matrix. Stops as soon as flagStart = 2
            if re.search(r'Region\:\s+\d', line):
              regionNumber = line.split()[-1]
            if re.search(r'\d+.\d+E', line):
              line = re.sub(r'[\|]', ' ', line)
              line = line.split()
              for group in range(1, self.Ngroups + 1):
                myMatrixDict[line[0]][str(regionNumber)][str(group)][str(
                    mpi)] = line[group]
    return myMatrixDict

  def mapColumns(self, line, count):
    """
      Allocates a column number relative to the reaction rates.
      @ In, line, string
      @ In, count, interger, counts the column position
      @ Out, numbering, dictionary, key: reaction rate name, value: column number
    """
    numbering = {}
    line = re.sub(r'averaged', r'', line)
    line = re.sub(r'fis. ', r'', line)
    line = re.sub(r'[\|]', ' ', line)
    parameterNames = line.split()
    for param in parameterNames:
      numbering[param] = count
      count += 1
    return numbering

  def locateXYandGroups(self, IDlist):
    """
      Locates what the position number of the x, y, z coordinates and the first energy group are in the INSTANT csv output file.
      @ In, IDlist, list, list of all the parameter in the csv output
      @ Out, xPositionInList, integer, position of the parameter x in the list
      @ Out, yPositionInList, integer, position of the parameter y in the list
      @ Out, zPositionInList, integer, position of the parameter z in the list
      @ Out, firstGroupPositionInList, integer, position of the first energy parameter
    """
    xPositionInList = None
    yPositionInList = None
    zPositionInList = None
    firstGroupPositionInList = None
    for i in range(len(IDlist)):
      if IDlist[i] == 'X':
        xPositionInList = i
      if IDlist[i] == 'Y':
        yPositionInList = i
      if IDlist[i] == 'Z':
        zPositionInList = i
      if IDlist[i] == '1':
        group1PositionInList = i
        break
    return xPositionInList, yPositionInList, zPositionInList, group1PositionInList

  def getFluxInfo(self, input):
    """
      Reads the INSTANT csv file to get the flux info relative to each region and each group.
      The flux info are also provided for each material.
      @ In, input, string, file name
      @ Out, fluxLabelList, list, labels relative to flux
      @ Out, fluxList, list, flux values
      @ Out, matFluxLabelList, list, labels relative to material flux
      @ Out, matFluxList, list, material flux values
    """
    IDlist = []
    fluxLabelList = []
    fluxList = []
    matFluxLabelList = []
    matFluxList = []
    flagFlux = 0
    countTimeStep = 0
    with open(os.path.join(self.workingDir, input), 'r') as outfile:
      for line in outfile:
        if re.search(
            r'PHISICS',
            line):  # first line of the csv file, the time info is here
          flagFlux = 0
          countTimeStep = countTimeStep + 1
        elif re.search(
            r'FLUX BY CELLS',
            line):  # beginning of the cell flux info in the csv file
          flagFlux = 1
        elif re.search(
            r'FLUX BY MATERIAL', line
        ):  # end of the cell flux info, beginning of the material flux info
          flagFlux = 2
        elif re.search(r'FLUX BY SUBDOMAIN',
                       line):  # end of the material flux info.
          flagFlux = 3

        if flagFlux == 1:
          if re.search(r'ID\s+,\s+ID', line):
            line = re.sub(r' ', r'', line)
            IDlist = line.split(',')
            xPosition, yPosition, zPosition, group1Position = self.locateXYandGroups(
                IDlist)
            IDlist.remove('\n')
          stringIsNumber = self.isNumber(line.split(','))
          if stringIsNumber:
            line = re.sub(
                r' ', r'', line
            )  # removes the spaces, in order to have only comma-separated values
            line = line.rstrip(
            )  # removes the newline, in order to have only comma-separated values
            for g in range(1, self.Ngroups + 1):
              if countTimeStep == 1:  # the labels are the same for each time step
                fluxLabelList.append('flux' + '|' + 'cell' +
                                     line.split(',')[0] + '|' + 'gr' + str(g))
              fluxList.append(line.split(',')[group1Position + g - 1])
        if flagFlux == 2:
          stringIsNumber = self.isNumber(line.split(','))
          if stringIsNumber:
            line = re.sub(r' ', r'', line)
            line = re.sub(r'\n', r'', line)
            for g in range(1, self.Ngroups + 1):
              if countTimeStep == 1:
                matFluxLabelList.append('flux' + '|' + 'mat' + line.split(',')
                                        [0] + '|' + 'gr' + str(g))
              matFluxList.append(line.split(',')[g])
    return fluxLabelList, fluxList, matFluxLabelList, matFluxList

  def getFluxInfoPhiRel(self, input):
    """
      Gets the flux info by cell and material if the PHISICS/RELAP coupled version is used.
      @ In, input, string, csv output
      @ Out, labelList, list, list of labels
      @ Out, varList, list, lidt of values relative to the labels
    """
    IDlist = []
    labelList = []
    varList = []
    flagFlux = 0
    countTimeStep = 0
    with open(os.path.join(self.workingDir, input), 'r') as outfile:
      for line in outfile:
        if re.search(
            r'PHISICS',
            line):  # first line of the csv file, the time info is here
          flagFlux = 0
          countTimeStep = countTimeStep + 1
        elif re.search(
            r'FLUX BY CELLS',
            line):  # beginning of the cell flux info in the csv file
          flagFlux = 1
        elif re.search(
            r'POWER DENSITY BY CELLS', line
        ):  # end of the power density info, beginning of the material flux info
          flagFlux = 2
        elif re.search(r'FLUX BY SUBDOMAIN',
                       line):  # end of the material flux info.
          flagFlux = 3

        if flagFlux == 1:
          if re.search(r'ID\s+,\s+ID', line):
            line = re.sub(r' ', r'', line)
            IDlist = line.split(',')
            xPosition, yPosition, zPosition, group1Position = self.locateXYandGroups(
                IDlist)
            IDlist.remove('\n')
          stringIsNumber = self.isNumber(line.split(','))
          if stringIsNumber:
            line = re.sub(
                r' ', r'', line
            )  # removes the spaces, in order to have only comma-separated values
            line = line.rstrip(
            )  # removes the newline, in order to have only comma-separated values
            for g in range(1, self.Ngroups + 1):
              if countTimeStep == 1:  # the labels are the same for each time step
                labelList.append('flux' + '|' + 'cell' + line.split(',')[0] +
                                 '|' + 'gr' + str(g))
              varList.append(line.split(',')[group1Position + g - 1])

        if flagFlux == 2:
          stringIsNumber = self.isNumber(line.split(','))
          if stringIsNumber:
            line = re.sub(
                r' ', r'', line
            )  # removes the spaces, in order to have only comma-separated values
            line = line.rstrip(
            )  # removes the newline, in order to have only comma-separated values
            for g in range(1, self.Ngroups + 1):
              if countTimeStep == 1:
                labelList.append('flux' + '|' + 'mat' + line.split(',')[0] +
                                 '|' + 'gr' + str(g))
              varList.append(line.split(',')[g])
    return labelList, varList

  def getDepInfo(self, timeStepIndex, matchedTimeSteps, numberOfMPI):
    """
      Reads the INSTANT csv file to get the material density info relative to depletion.
      @ In, timeStepIndex, integer, timestep number
      @ In, matchedTimeSteps, list, list of time steps considered
      @ In, numberOfMPI, integer, number of MPI used
      @ Out, depLabelList, list, list of labels relative to depletion
      @ Out, depList, list, list of values relative to depletion
    """
    materialList = []
    depLabelList = []
    depList = []
    timeStepList = []
    matList = []
    for mpi in range(numberOfMPI):
      with open(
          os.path.join(self.workingDir, self.mrtauOutputFileMPI[mpi]),
          'r') as outfile:
        for line in outfile:
          line = re.sub(
              r' ', r'', line
          )  # remove all spaces in a line. comma separated lines are parsed below
          if re.search(r'TIME', line):
            line = line.rstrip()
            self.isotopeList = line.split(',')
          if re.search(r'Material', line):
            materialList = line.split(',')
            matList.append(line.split(',')[1])
          stringIsFloatNumber = self.isFloatNumber(line.split(','))
          if stringIsFloatNumber:
            line = re.sub(r'\n', r'', line)
            if float(line.split(',')[0]) == float(matchedTimeSteps[timeStepIndex]):
              for i in range(1, len(self.isotopeList)):
                timeStepList.append(line.split(',')[0])
                depLabelList.append(
                    'dep' + '|' + materialList[1] + '|' + self.isotopeList[i])
                depList.append(line.split(',')[i])
    return depLabelList, depList, timeStepList, matList

  def getDepInfoMrtau(self, timeStepIndex, matchedTimeSteps, atomsInp):
    """
      Reads the MRTAU csv file to get the material density info relative to depletion.
      @ In, timeStepIndex, integer, timestep number
      @ In, matchedTimeSteps, list, list of time steps considered
      @ In, atomsInp, string, path to file pointed by the node <atoms_csv> in the lib path file
      @ Out, depLabelList, depList
    """
    depList = []
    with open(os.path.join(self.workingDir, atomsInp), 'r') as outfile:
      for line in outfile:
        line = re.sub(r' ', r'', line)
        stringIsFloatNumber = self.isFloatNumber(line.split(','))
        if stringIsFloatNumber and float(line.split(',')[0]) == float(
            matchedTimeSteps[timeStepIndex]):
          line = re.sub(r'\n', r'', line)
          for i in range(1, len(self.isotopeListMrtau) + 1):
            depList.append(line.split(',')[i])
    return depList

  def numberOfMediaUsed(self, mpi, numberOfMPI):
    """
      Finds the number of media used in a given INSTANT output.
      @ In, mpi, integer, MPI number considered
      @ In, numberOfMPI, integer, total number of MPI user-selected
      @ Out, mediaUsed, integer, number of media treated in one INSTANT MPI output
    """
    test = 0
    count = None
    if not (self.phisicsRelap and mpi == numberOfMPI - 1):
      outfile = open(
          os.path.join(self.workingDir, self.instantOutputFileMPI[mpi]))
      count = sum([
          1 for line in outfile.readlines()
          if re.search(r'Medium\s+\d+\s+used', line)
      ])
    return count

  def getCPUtime(self, numberOfMPI):
    """
      Gets the PHISICS CPU time.
      @ In, numberOfMPI, integer, number of MPI user-selected
      @ Out, cpuTime, list, cpu time (string) in a list
    """
    cpuTimes = []
    for mpi in range(numberOfMPI):
      with open(os.path.join(self.workingDir,
                             self.instantOutputFileMPI[mpi])) as outfile:
        for line in outfile:
          if re.search(r'CPU\s+time\s+\(min\)',
                       line) and self.phisicsRelap is False:
            cpuTimes.append(line.strip().split(' ')[-1])
          if self.phisicsRelap:
            return 0.
    cpuTime = sum([float(elm) for elm in cpuTimes])
    return [cpuTime]

  def getCPUtimeMrtau(self, mrtauFile):
    """
      Gets the MRTAU CPU time.
      @ In, mrtauFile, string, MRTAU file parsed
      @ Out, getCPUtimeMrtau, list, contains one element, cpu time (string)
    """
    with open(os.path.join(self.workingDir, mrtauFile)) as outfile:
      for line in outfile:
        if re.search(r'Cpu\s+Time\s+min', line) and self.phisicsRelap is False:
          return [line.strip().split(' ')[-1]]
        if self.phisicsRelap:
          return 0.

  def locateMaterialInFile(self, numberOfMPI):
    """
      Finds the material names in a given INSTANT output file.
      @ In, numberOfMPI, integer, number of MPI user-selected
      @ Out, materialsDict, dictionary, dictionary listing the media treated in a given mpi output.
              format: {MPI-0:{1:fuel1_1, 2:fuel1_5, 3:{fuel1_7}}, MPI-2:{1:fuel1_2, 2:fuel1_3, 3:fuel1_4, 4:fuel1_6}}
    """
    materialsDict = {}
    for mpi in range(numberOfMPI):
      count = 0
      materialsDict['MPI-' + str(mpi)] = {}
      mediaUsed = self.numberOfMediaUsed(mpi, numberOfMPI)
      if not (self.phisicsRelap and mpi == numberOfMPI - 1
              and numberOfMPI > 1):
        with open(
            os.path.join(self.workingDir,
                         self.instantOutputFileMPI[mpi])) as outfile:
          for line in outfile:
            if re.search(r'Density spatial moment', line):
              count = count + 1
              matLine = list(filter(None, line.split(' ')))
              materialsDict['MPI-' + str(mpi)][count] = matLine[
                  matLine.index('Material') + 1]
              if count == mediaUsed:
                break
    return materialsDict

  def getDecayHeat(self, timeStepIndex, matchedTimeSteps, decayHeatFlag):
    """
      Reads the main output file to get the decay heat.
      @ In, timeStepIndex, integer, number of the timestep considered
      @ In, matchedTimeSteps, list, list of time steps considered
      @ In, decayHeatFlag, integer, 1 means no decay heat, 2 decay heat in KW, 3 decay heat in MeV/s
      @ Out, decayLabelList, list, list of the decay labels (under the format fuel|isotope)
      @ Out, decayList, list, list of the decay heat values
    """
    materialList = []
    decayLabelList = []
    decayList = []
    timeStepList = []

    if decayHeatFlag == 1:
      decayLabelList = ['decayHeat']
      decayList = 0.
      return decayLabelList, decayList
    else:
      for mpi in range(len(self.instantOutputFileMPI)):
        decayFlag = 0
        breakFlag = 0
        matLineFlag = 0
        materialCounter = 0
        with open(
            os.path.join(self.workingDir,
                         self.instantOutputFileMPI[mpi])) as outfile:
          for line in outfile:
            if re.search(r'INDIVIDUAL DECAY HEAT BLOCK', line):
              decayFlag = 1
              materialCounter = materialCounter + 1
            if re.search(r'CUMULATIVE DECAY HEAT BLOCK', line):
              decayFlag = 0
            if re.search(r'BURNUP OUTPUT', line):
              breakFlag = 1
            if decayFlag == 1 and breakFlag == 0:
              line = line.rstrip()
              decayLine = list(filter(None, line.split(' ')))
              if decayLine != []:
                stringIsFloatNumber = self.isFloatNumber(decayLine)
              if stringIsFloatNumber and decayLine != []:
                if abs(
                    float(decayLine[0]) -
                    float(matchedTimeSteps[timeStepIndex])) < 1.0e-10:
                  for i in range(1, len(self.isotopeList)):
                    decayLabelList.append(
                        'decay' + '|' +
                        self.materialsDict['MPI-' + str(mpi)][materialCounter]
                        + '|' + self.isotopeList[i]
                    )  #index should be staggered with index in next lin (to test)
                    decayList.append(decayLine[i])
            if breakFlag == 1:
              break
      return decayLabelList, decayList

  def getDecayHeatMrtau(self, timeStepIndex, matchedTimeSteps,
                        decayHeatOutput):
    """
      Gets the decay heat from the standalone MRTAU output.
      @ In, timeStepIndex, integer, timestep number
      @ In, matchedTimeSteps, list, list of time steps considered
      @ In, decayHeatOutput, string, decay hat output file name
      @ Out, decayListMrtau, list, list of the decay values from MRTAU
    """
    decayFlag = 0
    breakFlag = 0
    self.decayLabelListMrtau = []
    self.numDensityLabelListMrtau = []
    decayListMrtau = []
    with open(os.path.join(self.workingDir, decayHeatOutput), 'r') as outfile:
      for line in outfile:
        if re.search(r'INDIVIDUAL DECAY HEAT BLOCK', line):
          decayFlag = 1
        if re.search(r'CUMULATIVE DECAY HEAT BLOCK', line):
          breakFlag = 1
        if decayFlag == 1 and breakFlag == 0:
          line = line.rstrip()
          decayLine = list(filter(None, line.split(' ')))
          if decayLine != []:
            stringIsFloatNumber = self.isFloatNumber(decayLine)
          if stringIsFloatNumber and decayLine != []:
            if float(decayLine[0]) == float(matchedTimeSteps[timeStepIndex]):
              for i in range(0, len(self.isotopeListMrtau)):
                self.numDensityLabelListMrtau.append(
                    'numDensity' + '|' + self.isotopeListMrtau[i])
                self.decayLabelListMrtau.append(
                    'decay' + '|' + self.isotopeListMrtau[i])
                decayListMrtau.append(decayLine[i + 1])
          if breakFlag == 1:
            break
    return decayListMrtau

  def getBurnUp(self, timeStepIndex, matchedTimeSteps):
    """
      Reads the main output file to get the burn up.
      @ In, timeStepIndex, integer, number of the timestep considered
      @ In, matchedTimeSteps, list, list of time steps considered
      @ Out, decayLabelList, list, list of the decay labels (under the format fuel|isotope)
      @ Out, decayList, list, list of the decay heat values
    """
    materialList = []
    buLabelList = []
    buList = []
    timeStepList = []
    for mpi in range(len(self.instantOutputFileMPI)):
      buFlag = 0
      breakFlag = 0
      matLineFlag = 0
      materialCounter = 0
      with open(os.path.join(self.workingDir,
                             self.instantOutputFileMPI[mpi])) as outfile:
        for line in outfile:
          if re.search(r'Burn Up \(GWd/MTHM\)', line):
            buFlag = 1
            materialCounter = materialCounter + 1
          if re.search(r'Mass \(kg\)', line):
            breakFlag = 1
          if buFlag == 1 and breakFlag == 0:
            line = line.rstrip()
            buLine = list(filter(None, line.split(' ')))
            if buLine != []:
              stringIsFloatNumber = self.isFloatNumber(buLine)
            if stringIsFloatNumber and buLine != []:
              if abs(
                  float(buLine[0]) - float(matchedTimeSteps[timeStepIndex])
              ) < 1.0e-10:
                buLabelList.append(
                    'bu' + '|' +
                    self.materialsDict['MPI-' + str(mpi)][materialCounter])
                buList.append(buLine[1])  # the burn up is on the second colum
          if breakFlag == 1:
            break
    return buLabelList, buList

  def getRRlist(self, instantDict):
    """
      Puts the reaction rate labels and the reaction rate values in a list.
      @ In, instantDict, dictionary
      @ Out, rrNames, list, list of parameter labels that are in the reaction rate matrix
      @ Out, rrValues, list, values relative to the labels, within the reaction rate matrix
    """
    rrNames = []
    rrValues = []
    if self.printSpatialRR:
      for param in self.paramList:
        for j in range(1, self.Ngroups + 1):
          for k in range(1, self.Nregions + 1):
            if param not in ['Group', 'Region']:
              rrNames.append(param + '|gr' + str(j) + '|reg' + str(k))
              rrValues.append(
                  instantDict.get('reactionRateInfo').get(str(j)).get(
                      str(k)).get(param))
    else:
      for param in self.paramList:
        if param not in ['Group', 'Region']:
          rrNames.append(param + '|Total')
          rrValues.append(instantDict.get('reactionRateInfo').get(param))
    if 'Group' in rrNames:
      rrNames.remove('Group')
    return rrNames, rrValues

  def phisicsTimeStepData(self, instantDict, timeStepIndex, matchedTimeSteps):
    """
      Return PHISICS data
      @ In, InstantDict, dictionary, contains all the values collected from INSTANT output
      @ In, timeStepIndex, integer, timestep number
      @ In, matchedTimeSteps, list, list of time steps considered
      @ Out, headers, list, the list of variables
      @ Out, snapshoot, np.array, the values for this timestep (timeStepIndex)
    """
    rrNames, rrValues = self.getRRlist(instantDict)
    if self.phisicsRelap:
      headers = ['timeMrTau'] + ['keff'] + ['errorKeff'] + rrNames + instantDict.get('fluxLabelList')
      headers += instantDict.get('powerDensLabelList') + instantDict.get('depLabelList') + instantDict.get('decayLabelList')
      headers += instantDict.get('xsLabelList') + ['cpuTime'] + instantDict.get('buLabelList')
      snapshoot = np.asarray([str(matchedTimeSteps[timeStepIndex])] + instantDict.get('keff')
                             + instantDict.get('errorKeff') + rrValues +instantDict.get('fluxList') + instantDict.get('powerDensList')
                             + instantDict.get('depList') + instantDict.get('decayList') +instantDict.get('xsList')
                             + [instantDict.get('cpuTime')] + instantDict.get('buList'),dtype=float)
    else:
      headers  = ['timeMrTau'] + ['keff'] + ['errorKeff'] + rrNames
      headers += instantDict.get('fluxLabelList') + instantDict.get('matFluxLabelList')
      headers += instantDict.get('depLabelList')+ instantDict.get('decayLabelList')
      headers += instantDict.get('xsLabelList')+ ['cpuTime']+ instantDict.get('buLabelList')
      snapshoot = np.asarray([str(matchedTimeSteps[timeStepIndex])] + instantDict.get('keff')
                             + instantDict.get('errorKeff') + rrValues + instantDict.get('fluxList')
                             + instantDict.get('matFluxList') + instantDict.get('depList') + instantDict.get('decayList')
                             + instantDict.get('xsList') + instantDict.get('cpuTime') + instantDict.get('buList'), dtype=float)
    return headers, snapshoot

  def mrtauTimeStepData(self, mrtauDict):
    """
      Return mrtau data
      @ In, mrtauDict, dictionary, contains all the values collected from MRTAU output
      @ Out, headers, list, the list of variables
      @ Out, snapshoot, np.array, the values of the variables
    """
    headers = ['timeMrTau'] + self.numDensityLabelListMrtau + self.decayLabelListMrtau
    snapshoot = np.asarray([str(mrtauDict.get('mrtauTimeSteps')[mrtauDict.get('timeStepIndex')])]
                           + mrtauDict.get('depList') + mrtauDict.get('decayHeatMrtau'),dtype=float)
    return headers, snapshoot

  def returnData(self):
    """
      Method to return the data in a dictionary
      @ In, None
      @ Out, self.data, dict, the dictionary containing the data {var1:array,var2:array,etc}
    """
    return self.data
