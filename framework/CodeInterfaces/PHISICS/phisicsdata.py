"""
Created on July 25th, 2017
@author: rouxpn
"""
import os
import re 
import csv
import xml.etree.ElementTree as ET 
from decimal import Decimal 

class phisicsdata():
  """
    This class parses the phisics output of interest. The output of interest are placed in the a csv file. 
  """
  def __init__(self, phisicsDataDict):
    """
      read the phisics output
      @ In, phisicsDataDict, dictionary, dictionary of variables passed by the interface
      @ Out, None 
    """
    self.phisicsRelap = phisicsDataDict['phiRel']               # boolean: True means phisics relap coupled, False means phisics standalone 
    self.printSpatialRR = phisicsDataDict['printSpatialRR']     # boolean: True means spatial RR are printed, False means spatial RR are not printed 
    self.printSpatialFlux = phisicsDataDict['printSpatialFlux'] # boolean: True means spatial fluxes are printed, False means spatial fluxes are not printed
    self.workingDir = phisicsDataDict['workingDir']             # string: working directory
    self.mrtauOutputFileMPI   = []
    instantDict          = {}
    mrtauDict            = {}    
    markerList           = ['Fission matrices of','Scattering matrices of','Multigroup solver ended!'] 
    
    self.instantOutputFileMPI,self.mrtauOutputFileMPI = self.fileOutName(phisicsDataDict)
    mrtauOutput = [self.mrtauOutputFileMPI,phisicsDataDict['mrtauFileNameDict']['decay_heat']]
    cpuTime            = self.getCPUtime(phisicsDataDict['numberOfMPI'])
    if self.phisicsRelap is False:  
      instantOutput       = ['Dpl_INSTANT_'+phisicsDataDict['jobTitle']+'_flux_mat.csv', 'scaled_xs.xml']      
    if self.phisicsRelap is True:  
      instantOutput       = ['PHISICS_RELAP5_'+phisicsDataDict['jobTitle']+'_flux_sub.csv','PHISICS_RELAP5_'+phisicsDataDict['jobTitle']+'_power_dens_sub.csv','scaled_xs.xml']     
    
    self.materialsDict = self.locateMaterialInFile(phisicsDataDict['numberOfMPI'])      
    self.perturbationNumber = self.getPertNumber()
    self.cleanUp(phisicsDataDict['jobTitle'])
    self.getNumberOfGroups()
    self.getNumberOfRegions()
    
    if phisicsDataDict['mrtauStandAlone'] == False:   # mrtau and instant are coupled
      mrtauTimeSteps      = self.getMrtauInstantTimeSteps()
      instantTimeSteps    = self.getInstantTimeSteps(instantOutput[0])
      XSlabelList, XSlist = self.getAbsoluteXS(instantOutput[1])
    if phisicsDataDict['mrtauStandAlone'] == True:    # mrtau is standalone mode
      mrtauTimeSteps = self.getMrtauTimeSteps()
      self.getMrtauIsotopeList()
      
    for timeStepIndex in xrange (0,len(mrtauTimeSteps)): 
      if phisicsDataDict['mrtauStandAlone'] == False:
        keff, errorKeff      = self.getKeff()
        reactionRateInfoMPI  = self.getReactionRates(phisicsDataDict['numberOfMPI'])
        reactionRateInfo     = self.summedDictValues(reactionRateInfoMPI,'reactionRates',self.numberingRR)
        if self.printSpatialRR is True: 
          fissionMatrixInfoMPI = self.getMatrix(markerList[0],markerList[1],'FissionMatrix',phisicsDataDict['numberOfMPI'])
          fissionMatrixInfo    = self.summedDictValues(fissionMatrixInfoMPI,'FissionMatrix',self.matrixNumbering)
        else: 
          fissionMatrixInfo = {}
        if self.printSpatialFlux is True:
          if self.phisicsRelap is False:
            fluxLabelList, fluxList, matFluxLabelList, matFluxList = self.getFluxInfo(instantOutput[0])
          if self.phisicsRelap is True: 
            fluxLabelList, fluxList           = self.getFluxInfoPhiRel(instantOutput[0])
            powerDensLabelList, powerDensList = self.getFluxInfoPhiRel(instantOutput[1])
        if self.printSpatialFlux is False: 
          fluxLabelList = ['fluxLabel']
          fluxList = [0.0000E+00]
          matFluxLabelList = ['matFluxLabelList']
          matFluxList = [0.0000E+00]
          powerDensLabelList = ['powerDensLabelList']
          powerDensList = [0.0000E+00]
        depLabelList, depList, timeStepList, matList = self.getDepInfo(timeStepIndex, mrtauTimeSteps, phisicsDataDict['numberOfMPI'])
        decayLabelList, decayList                    = self.getDecayHeat(timeStepIndex,mrtauTimeSteps)
        
        instantDict['keff']              = keff
        instantDict['errorKeff']         = errorKeff
        instantDict['reactionRateInfo']  = reactionRateInfo
        instantDict['fissionMatrixInfo'] = fissionMatrixInfo
        instantDict['workingDir']        = self.workingDir
        instantDict['fluxLabelList']     = fluxLabelList 
        instantDict['fluxList']          = fluxList
        instantDict['depLabelList']      = depLabelList
        instantDict['depList']           = depList
        instantDict['timeStepList']      = timeStepList
        instantDict['decayLabelList']    = decayLabelList
        instantDict['decayList']         = decayList
        instantDict['timeStepIndex']     = timeStepIndex
        instantDict['mrtauTimeSteps']    = mrtauTimeSteps
        instantDict['XSlabelList']       = XSlabelList
        instantDict['XSlist']            = XSlist
        instantDict['cpuTime']           = cpuTime
        if self.phisicsRelap is True:
          instantDict['powerDensList']      = powerDensList
          instantDict['powerDensLabelList'] = powerDensLabelList
        if self.phisicsRelap is False:
          instantDict['matFluxLabelList']  = matFluxLabelList
          instantDict['matFluxList']       = matFluxList

        self.writeCSV(instantDict,timeStepIndex,mrtauTimeSteps,phisicsDataDict['jobTitle'])
      
      if phisicsDataDict['mrtauStandAlone'] == True:
        decayHeatMrtau = self.getDecayHeatMrtau(timeStepIndex, mrtauTimeSteps)
        depList        = self.getDepInfoMrtau(timeStepIndex, mrtauTimeSteps)
        
        mrtauDict['workingDir']     = self.workingDir
        mrtauDict['decayHeatMrtau'] = decayHeatMrtau 
        mrtauDict['depList']        = depList
        mrtauDict['timeStepIndex']  = timeStepIndex
        mrtauDict['mrtauTimeSteps'] = mrtauTimeSteps
        self.writeMrtauCSV(mrtauDict)
  
  def summedListValues(self,list):
    """
      sums up values from a given list
      @ In, list, list of intergers or floats
      @ Out, sum, integer or float, sum of the values given in input 
    """
    sum = 0
    for i in xrange (0,len(list)):
      sum = sum + float(list[i])
    return sum 
  
  def summedDictValues(self,dict,typeOfParameter,parameterList):
    """
      Sums up the values from a given dictionary 
      @ In, dict, dict, nested dictionaries of intergers or floats
      @ In, typeOfParameter, string, either 'reactionRates' or 'FissionMatrix' 
      @ In, parameterList, list, list of the parameters parsed
      @ Out, summedDict, dict, dictionary of integer or float 
    """
    summedDict = self.declareDict(typeOfParameter,parameterList)
    if self.printSpatialRR is False:
      for parameter,mpi in dict.iteritems():
        sum = 0 
        for mpi,value in mpi.iteritems():
          sum = '%.15E' % Decimal(float(sum) + float(value))
        summedDict[parameter] = sum 
    if self.printSpatialRR is True: 
      for group,region in dict.iteritems():  
        for region,parameter in region.iteritems():
          for parameter,mpi in parameter.iteritems():
            sum = 0
            for mpi,value in mpi.iteritems():
              sum = '%.15E' % Decimal(float(sum) + float(value))
              #print typeOfParameter
              #print value 
            summedDict[group][region][parameter] = sum
    #print typeOfParameter
    #print summedDict
    return summedDict
      
  def fileOutName(self,phisicsDataDict):
    """
      Puts in a list the instant output file names based on the number of MPI. 
      The format is title.o-MPI
      @ In, phisicsDataDict, dictionary, dictionary of variables passed by the interface
      @ Out, instantOutputFileMPI, list, list of instant output file names 
      @ Out, mrtauOutputFileMPI, list, list of mrtau output file names 
    """
    instantMPI = []
    mrtauMPI = []
    for mpi in xrange (0,phisicsDataDict['numberOfMPI']):
      mrtauMPI.append(os.path.join(self.workingDir,phisicsDataDict['mrtauFileNameDict']['atoms_csv'].split('.')[0]+'-'+str(mpi)+'.csv')) # Mrtau files (numbers.csv)
      if self.phisicsRelap is False:    # instant files, no coupling with relap 
          instantMPI.append(phisicsDataDict['instantOutput']+'-'+str(mpi))
      if self.phisicsRelap is True:     # instant files, coupling with relap 
        if mpi == 0: 
          pass 
        else:
          instantMPI.append('INSTout-'+str(mpi))
    return instantMPI, mrtauMPI
   
  def cleanUp(self,jobTitle):
    """
      Removes the file that RAVEN reads for postprocessing 
      @ In, jobTitle, string, job title name 
      @ Out, None 
    """
    csvOutput = os.path.join(self.workingDir, jobTitle+'-'+self.perturbationNumber+'.csv')
    if os.path.isfile(csvOutput):
      os.remove(csvOutput) 
    with open (csvOutput,'wb'):
      pass
    
  def getAbsoluteXS(self,XSoutput):
    """
      Parses the absolute cross section output generated by PHISICS
      @ In, XSoutput, string, filename of the XML file containing the absolute perturbed XS
      @ Out, labelList, list of the XS labels
      @ Out, valueList, list of the XS 
    """
    labelList = []
    valueList = []
    tree = ET.parse(os.path.join(self.workingDir,XSoutput))
    root = tree.getroot()
    for materialXML in root.getiterator('library'):
      for isotopeXML in materialXML.getiterator('isotope'):
        reactionList = [j.tag for j in isotopeXML]
        for k in xrange (0, len(reactionList)):
          for groupXML in isotopeXML.getiterator(reactionList[k]):
            individualGroup = [x.strip() for x in groupXML.attrib.get('g').split(',')]
            individualGroupValues = [y.strip() for y in groupXML.text.split(',')]
            for position in xrange(0,len(individualGroup)):
              labelList.append(materialXML.attrib.get('lib_name')+'|'+isotopeXML.attrib.get('id')+'|'+reactionList[k]+'|'+individualGroup[position])
              valueList.append(individualGroupValues[position])
    return labelList, valueList 
    
  def removeSpaces(self, line):
    """
      Removes the spaces. It makes the word splitting cleaner. 
      @ In, line, string
      @ Out, line, string, same line without the blank spaces
    """
    line = re.sub(r' ',r'',line)
    return line 
    
  def getNumberOfGroups(self):
    """
      Gives the number of energy groups used in the Phisics simulations  
      @ In, None 
      @ Out, None 
    """
    self.Ngroups = 0 
    with open(os.path.join(self.workingDir,self.instantOutputFileMPI[0]), 'r') as outfile:
      for line in outfile :
        if re.search(r'Number of elements of group', line) :
          self.Ngroups = line.split()[-3]  # pick the last occurence (last group), which is also the number of groups
        if re.search(r'Number of element  types', line) :
          break    
  
  def getNumberOfRegions(self,):
    """
      Gives the number of spatial regions used in the Phisics simulations  
      @ Out, None 
    """
    flagStart, count  =  0, 0
    subdomainList = []
    with open(os.path.join(self.workingDir, self.instantOutputFileMPI[0]), 'r') as outfile:
      for line in outfile :
        if re.search(r'Subdomain volumes', line) :
          flagStart = 1
        if re.search(r'Balance report for the primal solution', line) :
          flagStart = 2
        if flagStart == 1:
          stringIsNumber = self.isNumber(line.split())
          if stringIsNumber is True: 
            count = count + 1 
            subdomainList.append(line.split()[0])
        if flagStart == 2: 
          break            
    self.Nregions = count
    return 
  
  def getPertNumber (self):
    """
      Gets the current perturbation number
      @ Out, pertNumber, string, perturbation number in a string format 
    """
    splitWorkDir = self.workingDir.split('/')
    pertNumber = splitWorkDir[-1]
    return pertNumber 
  
  def removeTheZeroTimeStep(self,timeSteps):
    """
      Removes the first time step, t=0, to make the number of time steps match with the number of time steps in Instant
      @ In, timeSteps, list, list of the time steps in mrtau
      @ Out, timeSteps, list, time steps in mrtau withtout the first time step
    """
    timeSteps.pop(0)
    return timeSteps 
    
  def getMrtauInstantTimeSteps(self):
    """
      Gets the time steps in the coupled mrtau instant output
      @ In, mrtauCSVOutput, string, mrtau CSV output filename
      @ Out, timeSteps, list 
    """
    count = 0
    timeSteps = []
    with open(os.path.join(self.workingDir,self.mrtauOutputFileMPI[0]), 'r') as outfile:
      for line in outfile :
        if re.search(r'Material',line): 
          count = count + 1 
        if count == 1:
          stringIsFloatNumber = self.isFloatNumber(line.split(','))
          if stringIsFloatNumber is True:
            timeSteps.append(line.split(',')[0])
        if count > 1:
          break 
    timeSteps = self.removeTheZeroTimeStep(timeSteps) 
    return timeSteps
    
  def getMrtauTimeSteps(self):
    """
      Gets the time steps in the mrtau standalone output 
      @ In, None 
      @ Out, timeSteps, list 
    """
    timeSteps = []
    with open(os.path.join(self.workingDir,self.mrtauOutputFileMPI[0]), 'r') as outfile:
      for line in outfile : 
        stringIsFloatNumber = self.isFloatNumber(line.split(','))
        if stringIsFloatNumber is True:
          timeSteps.append(line.split(',')[0])
    return timeSteps
    
  def getMrtauIsotopeList(self,mrtauCSVOutput):
    """
      Gets the isotope in the mrtau standalone output 
      @ In, None 
      @ Out, None 
    """
    self.isotopeListMrtau = []
    with open(os.path.join(self.workingDir,self.mrtauOutputFileMPI[0]), 'r') as outfile:
      for line in outfile : 
        if re.search(r'TIME',line):
          line = re.sub(r'TIME\s+\(days\)',r'',line)
          line = re.sub(r' ',r'',line)
          line = re.sub(r'\n',r'',line)
          self.isotopeListMrtau = filter(None,line.split(','))
          break
    
  def getInstantTimeSteps(self,input):
    """
      Gets the time steps in the Instant output 
      @ In, None 
      @ Out, timeSteps, list 
    """
    count = 0
    timeSteps = []
    with open(os.path.join(self.workingDir,input), 'r') as outfile:
      for line in outfile :
        if re.search(r'PHISICS',line): 
          timeSteps.append(line.split(',')[line.split(',').index('Time') + 1]) 
    return timeSteps

  def reduceDigits(self,mrtauTimeSteps): 
    """
      Reduces the number of digits in the mrtau time steps   
      @ In, mrtauTimeSteps, list
      @ Out, timeSteps, list 
    """
    timeSteps = []
    for i in xrange (0, len(mrtauTimeSteps)):
      timeSteps.append('%.6E' % (float(mrtauTimeSteps[i])))
    return timeSteps  
  
  def findUnits(self,line):
    """
      Returns the units in which the time steps are printed in the instant output
      @ In, line, string 
      @ Out, units, string
    """ 
    units = line.split(' ')[-3]
    return units
  
  def getKeff (self):
    """
      Gets the multiplication factor 
      @ Out, keff, string, floating number under string format of keff
      @ Out, errorKeff, string, floating number under string format of errorKeff
    """
    self.units = None 
    keff = []
    with open(os.path.join(self.workingDir,self.instantOutputFileMPI[0]), 'r') as outfile:
      for line in outfile :
        if re.search(r'Time Input Type', line):
          self.units = self.findUnits(line)
        if re.search(r'k-effective at the last', line) :
          keff = [line.split()[-1]]
        if re.search(r'error for the eigenvalue', line) :
            errorKeff = [line.split()[-1]]
    return keff, errorKeff

  def isNumber(self,line):
    """
      Checks if a string is an integer
      @ In, line, list, list of strings
      @ Out, Boolean, True if integer, False otherwise  
    """
    if line != []:
      try: 
        int(line[0])
        return True
      except ValueError:
        return False
        
  def isFloatNumber(self,line):
    """
      Checks if a string is a float number 
      @ In, line, list, list of strings
      @ Out, Boolean, True if integer, False otherwise 
    """
    if line != []:
      try: 
        float(line[0])
        return True
      except ValueError:
        return False
   
  def convertInDays(self,instantTimeSteps): 
    """
      Converts the Instant time steps (seconds) into days  
      @ In, instantTimeSteps, list
      @ Out, timeSteps, list 
    """
    timeSteps = []
    for i in xrange (0, len(instantTimeSteps)):
      timeSteps.append('%.6E' % (float(instantTimeSteps[i]) / (24 * 60 * 60)))
    return timeSteps 
   
  def declareDict(self,typeOfParameters,numbering):
    """
      Declares the RR dictionary  
      @ In, typeOfparameter, string, either 'reaction rate' or 'FissionMatrix', to select which matrix is to be parsed
      @ In, numbering, dict, parameter number (value) and column number (key)
      @ Out, dict, dictionary, dictionary with keys, and empty values
    """
    dict = {}
    if self.printSpatialRR is True:
      for i in xrange (1,int(self.Ngroups) + 1):
        dict[str(i)] = {}
        for j in xrange (1,int(self.Nregions) + 1):
          dict[str(i)][str(j)] = {}
          for k in xrange (0, len(numbering)):
            if numbering.keys()[k] == 'Group': 
              pass
            else: 
              dict[str(i)][str(j)][numbering.keys()[k]] = {} 
    if self.printSpatialRR is False: 
      for k in xrange (0, len(numbering)):
        if numbering.keys()[k] == 'Group': 
          pass
        else: 
          dict[numbering.keys()[k]] = {} 
    if typeOfParameters == 'reactionRates':   # puts the reaction rate labels in a list
      self.paramList = []
      for key in numbering.iterkeys():
        self.paramList.append(key)
        
    if typeOfParameters == 'FissionMatrix':   # puts the fission matrix labels in a list
      self.matrixList = []
      for key in numbering.iterkeys():
        self.matrixList.append(key) 
    return dict
        
  def getReactionRates(self, numberOfMPI):
    """
      Gets the reactions rates, power for each group in PHISICS 
      @ In, None 
      @ Out, ReactionRateDict, dictionary, dictionary containing the RR infos
    """
    flagStart,  count, countTimeStep = 0, 0, 0 
    for mpi in xrange (0,numberOfMPI): # parse all the segmented files 
      with open(os.path.join(self.workingDir,self.instantOutputFileMPI[mpi]), 'r') as outfile:
        for line in outfile :
          if re.search(r'averaged flux\s+power', line):
            if mpi == 0: 
              self.numberingRR = self.mapColumns(line,count)
              reactionRateDict = self.declareDict('reactionRates',self.numberingRR)  
            flagStart = 1 
            countTimeStep = countTimeStep + 1 
          if re.search(r'Fission matrices of all',line):
            flagStart = 2
          if flagStart == 1:
            if self.printSpatialRR is True:  # if the spatial reaction rates are printed 
              if re.search(r'\w+\s+\w+',line):
                line = re.sub(r'[\|]',' ',line)
                line = line.split()
                stringIsNumber = self.isNumber(line)
                if stringIsNumber == True :
                  for i in xrange (0,len(self.numberingRR)): 
                    groupNum  = line[0]
                    regionNum = line[1]
                    if self.paramList[i] == 'Group': 
                      pass 
                    else: 
                      reactionRateDict[groupNum][regionNum][self.paramList[i]][mpi] = line[self.numberingRR.get(self.paramList[i])] 
                      
            if self.printSpatialRR is False:  # if the spatial reaction rates are not printed 
              if re.search(r'Total',line):
                line = line.split()
                for i in xrange (0,len(self.numberingRR)):
                  if self.paramList[i] == 'Group' or self.paramList[i] == 'Region': 
                    pass
                  else:                
                    reactionRateDict[self.paramList[i]][mpi] = line[self.numberingRR.get(self.paramList[i])]
    if reactionRateDict != {}:
      return reactionRateDict  
    
  def getMatrix(self,startStringFlag,endStringFlag,typeOfMatrix,numberOfMPI):
    """
      Gets the fission matrix and scattering matrix in the PHISICS output 
      @ In, startStringFlag, string, string marker. 
      @ In, endStringFlag, string, string marker. 
      @ In, typeOfMatrix, string, allows to choose between fission matrix parsing or scattering matrix parsing
      @ In, numberOfMPI, interger, number of MPI user-defined 
      @ Out, matrixDict, dictionary
    """
    flagStart, count, countTimeStep = 0, 0, 0
    matrixDict = {}
    for mpi in xrange (0,numberOfMPI): # parse all the segmented files 
      with open(os.path.join(self.workingDir,self.instantOutputFileMPI[mpi]), 'r') as outfile:
        for line in outfile : 
          if re.search(startStringFlag, line):
            flagStart = 1 
            countTimeStep = countTimeStep + 1
          if re.search(endStringFlag,line):
            flagStart = 2
          if flagStart == 1 :
            if re.search(r'\s+1\s+2\s+3\s+4',line):
              line = re.sub(r'[\|]',' ',line)
              self.matrixNumbering = self.mapColumns(line,count)
              if mpi == 0: # declare the dictionary only at the first iteration
                matrixDict = self.declareDict(typeOfMatrix,self.matrixNumbering)
            if re.search(r'Region\:\s+\d',line): 
              regionNumber = line.split()[-1]
            if re.search(r'\d+.\d+E',line):
              line = re.sub(r'[\|]',' ',line)
              line = line.split()
              for i in xrange (1,len(self.matrixNumbering) + 1):
                print mpi
                matrixDict[line[0]][str(regionNumber)][str(i)][str(mpi)] = line[self.matrixNumbering.get(str(i)) + 1]    
    print 'matrixDict'
    print matrixDict
    return matrixDict
  
  def mapColumns(self,line,count): 
    """
      allocates a column number relative to the reaction rates 
      @ In, line, string
      @ In, count, interger, counts the column position
      @ Out, numbering, dictionary, key: RR name, value: column number 
    """
    numbering = {}
    line = re.sub(r'averaged',r'',line)
    line = re.sub(r'fis. ',r'',line)
    line = re.sub(r'[\|]',' ',line)
    parameterNames = line.split()
    for i in xrange(len(parameterNames)) :
      numbering[parameterNames[i]] = count
      count = count + 1
    return numbering 

  def locateXYandGroups(self,IDlist):
    """
      Locates what is the position number of the x, y, z coordinates and the first energy group in the Instant 
      csv output file. 
      @ In, IDlist, list, list of all the parameter in the csv output
      @ Out, xPositionInList, interger, position of the parameter x in the list
      @ Out, yPositionInList, interger, position of the parameter y in the list
      @ Out, zPositionInList, interger, position of the parameter z in the list
      @ Out, firstGroupPositionInList, interger, position of the first energy parameter 
    """
    xPositionInList = 0
    yPositionInList = 0
    zPositionInList = 0
    firstGroupPositionInList = 0
    for i in xrange (0,len(IDlist)):
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
    
  def getFluxInfo(self,input):
    """
      Reads the Instant CSV file to get the flux info relative to each region and each group
      The flux info are also provided for each material
      @ In, input, string, file name   
      @ Out, fluxLabelList, list
      @ Out, fluxList, list
      @ Out, matFluxLabelList, list 
      @ Out, matFluxList, list
    """
    IDlist = []
    fluxLabelList = []
    fluxList = []
    matFluxLabelList = []
    matFluxList = []
    flagFlux, countTimeStep = 0, 0 
    with open(os.path.join(self.workingDir,input), 'r') as outfile:
      for line in outfile :
        if re.search(r'PHISICS',line): 
          flagFlux = 0 
          countTimeStep = countTimeStep + 1
        if re.search(r'FLUX BY CELLS',line):    
          flagFlux = 1 
        if re.search(r'FLUX BY MATERIAL',line): 
          flagFlux = 2
        if re.search(r'FLUX BY SUBDOMAIN',line):
          flagFlux = 3
        
        if flagFlux == 1:
          if re.search(r'ID\s+,\s+ID',line):
            line = re.sub(r' ',r'',line)
            IDlist = line.split(',')
            xPosition, yPosition, zPosition, group1Position = self.locateXYandGroups(IDlist)
            IDlist.remove('\n')         
          stringIsNumber = self.isNumber(line.split(','))
          if stringIsNumber is True:
            line = re.sub(r' ',r'',line)
            line = re.sub(r'\n',r'',line)
            for g in xrange (1,int(self.Ngroups) + 1):
              if countTimeStep == 1: # the labels are the same for each time step
                fluxLabelList.append('flux'+'|'+'cell'+line.split(',')[0]+'|'+'gr'+str(g))
              fluxList.append(line.split(',')[group1Position + g - 1])
              
        if flagFlux == 2:        
          stringIsNumber = self.isNumber(line.split(','))
          if stringIsNumber is True:
            line = re.sub(r' ',r'',line)
            line = re.sub(r'\n',r'',line)
            for g in xrange (1,int(self.Ngroups) + 1):
              if countTimeStep == 1:
                matFluxLabelList.append('flux'+'|'+'mat'+line.split(',')[0]+'|'+'gr'+str(g))
              matFluxList.append(line.split(',')[g])
    return fluxLabelList, fluxList, matFluxLabelList, matFluxList
 
  def getFluxInfoPhiRel(self,input):
    """
      Gets the flux info by cell and material if the phisics/relap coupled version is used
      @ In, None 
      @ In, input, string, csv output 
      @ Out, labelList, list 
      @ Out, varList, list
    """
    IDlist = []
    labelList = []
    varList = []
    flagFlux, countTimeStep = 0, 0 
    with open(os.path.join(self.workingDir,input), 'r') as outfile:
      for line in outfile :
        if re.search(r'PHISICS',line): 
          flagFlux = 0 
          countTimeStep = countTimeStep + 1
        if re.search(r'FLUX BY CELLS',line):          
          flagFlux = 1 
        if re.search(r'POWER DENSITY BY CELLS',line): 
          flagFlux = 2
        if re.search(r'FLUX BY SUBDOMAIN',line):      
          flagFlux = 3 
        
        if flagFlux == 1:
          if re.search(r'ID\s+,\s+ID',line):
            line = re.sub(r' ',r'',line)
            IDlist = line.split(',')
            xPosition, yPosition, zPosition, group1Position = self.locateXYandGroups(IDlist)
            IDlist.remove('\n')         
          stringIsNumber = self.isNumber(line.split(','))
          if stringIsNumber is True:
            line = re.sub(r' ',r'',line)
            line = re.sub(r'\n',r'',line)
            for g in xrange (1,int(self.Ngroups) + 1):
              if countTimeStep == 1: # the labels are the same for each time step
                labelList.append('flux'+'|'+'cell'+line.split(',')[0]+'|'+'gr'+str(g))
              varList.append(line.split(',')[group1Position + g - 1])
              
        if flagFlux == 2:        
          stringIsNumber = self.isNumber(line.split(','))
          if stringIsNumber is True:
            line = re.sub(r' ',r'',line)
            line = re.sub(r'\n',r'',line)
            for g in xrange (1,int(self.Ngroups) + 1):
              if countTimeStep == 1:
                labelList.append('flux'+'|'+'mat'+line.split(',')[0]+'|'+'gr'+str(g))
              varList.append(line.split(',')[g])
    return labelList, varList
    
  def getMaterialList(self,line,matList):
    """
    Returns a list of all the problem materials
    @ In, matList, list
    @ Out, matList, list (appends additional material)
    """
    matList.append(line[1])
    return matList 
 
  def getDepInfo(self, timeStepIndex, matchedTimeSteps, numberOfMPI):
    """
      Reads the Instant CSV file to get the material density info relative to depletion
      @ In, timeStepIndex, integer, timestep number 
      @ In, matchedTimeSteps, list, list of time steps considered
      @ In, numberOfMPI, interger, number of MPI used 
      @ Out, depLabelList, list 
      @ Out, depList, list 
    """
    materialList = []
    depLabelList = []
    depList      = []
    timeStepList = []
    matList      = []
    for mpi in xrange(0,numberOfMPI):
      with open(os.path.join(self.workingDir,self.mrtauOutputFileMPI[mpi]), 'r') as outfile: 
        for line in outfile :
          line = self.removeSpaces(line)
          if re.search(r'TIME',line):
            line = re.sub(r'\n',r'',line)          
            self.isotopeList = line.split(',')
          if re.search(r'Material',line):
            materialList = line.split(',')
            matList = self.getMaterialList(line.split(','), matList)
          stringIsFloatNumber = self.isFloatNumber(line.split(',')) 
          if stringIsFloatNumber is True:
            line = re.sub(r'\n',r'',line)
            if (float(line.split(',')[0]) == float(matchedTimeSteps[timeStepIndex])):  
              for i in xrange (1,len(self.isotopeList)):
                timeStepList.append(line.split(',')[0])
                depLabelList.append('dep'+'|'+materialList[1]+'|'+self.isotopeList[i])
                depList.append(line.split(',')[i])
    return depLabelList, depList, timeStepList, matList
  
  def getDepInfoMrtau(self,timeStepIndex,matchedTimeSteps):
    """
      Reads the mrtau CSV file to get the material density info relative to depletion
      @ In, timeStepIndex, integer, timestep number 
      @ In, matchedTimeSteps, list, list of time steps considered
      @ Out, depLabelList, depList
    """
    depList = []
    with open(os.path.join(self.workingDir,self.mrtauOutputFileMPI[0]), 'r') as outfile:
      for line in outfile :
        line = self.removeSpaces(line)
        stringIsFloatNumber = self.isFloatNumber(line.split(',')) 
        if stringIsFloatNumber is True:
          if (float(line.split(',')[0]) == float(matchedTimeSteps[timeStepIndex])):
            line = re.sub(r'\n',r'',line)   
            for i in xrange (0,len(self.isotopeListMrtau)):
              depList.append(line.split(',')[i+1])
    return depList
    
  def findDecayHeat(self,line):
    """
      Determines whether the decay heat is printed
      @ In, line, string
      @ Out, isDecayHeatPrinted, string 
      @ Out, boolean, True if decay heat is printed, false otherwise 
    """
    DecayHeatUnits = None 
    isDecayHeatPrinted = line.split(' ')[-2]
    if isDecayHeatPrinted == 'YES': 
      DecayHeatUnits =  line.split(' ')[-1]
      return True, DecayHeatUnits
    else: 
      return False, DecayHeatUnits
  
  def numberOfMediaUsed(self,mpi,numberOfMPI):
    """
      Finds the number of media used in a given instant output. 
      @ In, mpi, integer, number of MPI user-selected
      @ Out, mediaUsed, integer, number of media treated in one Instant MPI output 
    """
    count = 0 
    if self.phisicsRelap is True and mpi >= numberOfMPI - 1 : pass 
    else: 
      with open(os.path.join(self.workingDir, self.instantOutputFileMPI[mpi])) as outfile:
        for line in outfile:
          if re.search(r'Medium\s+\d+\s+used',line):
            count = count + 1 
    return count

  def getCPUtime(self,numberOfMPI):
    """
      Gets the Phisics CPU time
      @ In, numberOfMPI, interger, number of MPI used
      @ Out, string, cpu time under string format 
    """
    cpuTimes = []
    for mpi in xrange (0,numberOfMPI):
      with open(os.path.join(self.workingDir,self.instantOutputFileMPI[mpi])) as outfile:
        for line in outfile: 
          if re.search(r'CPU\s+time\s+\(min\)',line) and self.phisicsRelap is False:  
            cpuTimes.append(line.strip().split(' ')[-1])
          if self.phisicsRelap is True:
            return 'na'
    cpuTime = self.summedListValues(cpuTimes)
    return [cpuTime]
          
  def locateMaterialInFile(self,numberOfMPI):
    """
      Finds the material names in a given Instant output file. 
      @ In, numberOfMPI, integer, number of MPI user-selected
      @ Out, materialDict, dictionary, dictionary listing the media treated in a given mpi output. 
              format: {MPI-0:{1:fuel1_1, 2:fuel1_5, 3:{fuel1_7}}, MPI-2:{1:fuel1_2, 2:fuel1_3, 3:fuel1_4, 4:fuel1_6}}
    """
    materialsDict = {}
    for mpi in xrange(0,numberOfMPI):
      count = 0 
      materialsDict['MPI-'+str(mpi)] = {}
      mediaUsed = self.numberOfMediaUsed(mpi,numberOfMPI)
      if self.phisicsRelap is True and mpi >= numberOfMPI - 1 : pass  
      else:
        with open(os.path.join(self.workingDir, self.instantOutputFileMPI[mpi])) as outfile:
          for line in outfile:
            if re.search(r'Density spatial moment',line):
              count = count + 1 
              matLine = filter(None, line.split(' '))
              materialsDict['MPI-'+str(mpi)][count] = matLine[matLine.index('Material') + 1]
              if count == mediaUsed:
                break 
    return materialsDict
   
  def getDecayHeat(self,timeStepIndex,matchedTimeSteps):
    """
      Read the main output file to get the decay heat
      @ In, timeStepIndex, integer, number of the timestep considered 
      @ In, matchedTimeSteps, list, list of time steps considered
      @ Out, decayLabelList, list, list of the decay labels (under the format fuel|isotope)
      @ Out, decayList, list, list of the decay values 
    """
    materialList = []
    decayLabelList = []
    decayList = []
    timeStepList = []
    for mpi in xrange(0,len(self.instantOutputFileMPI)):
      decayFlag, breakFlag, matLineFlag, materialCounter = 0, 0, 0, 0
      with open(os.path.join(self.workingDir, self.instantOutputFileMPI[mpi])) as outfile:
        for line in outfile:
          if re.search(r'Decay Heat computed',line) and mpi == 0 :
            self.isDecayHeatPrinted, self.decayHeatUnits = self.findDecayHeat(line)
            if self.isDecayHeatPrinted is False: 
              decayLabelList = ['decayHeat']
              decayList = 0
              return  decayLabelList,decayList
          if re.search(r'INDIVIDUAL DECAY HEAT BLOCK',line):  
            decayFlag = 1
            materialCounter = materialCounter + 1  
          if re.search(r'CUMULATIVE DECAY HEAT BLOCK',line):  
            decayFlag = 0
          if re.search(r'BURNUP OUTPUT',line):  
            breakFlag = 1
          if decayFlag == 1 and breakFlag == 0 :
            line = re.sub(r'\n',r'',line)
            decayLine = filter(None, line.split(' '))
            if decayLine != []: 
              stringIsFloatNumber = self.isFloatNumber(decayLine)
            if stringIsFloatNumber is True and decayLine != []:
              if (float(decayLine[0]) == float(matchedTimeSteps[timeStepIndex])):  
                for i in xrange (1,len(self.isotopeList)):
                  decayLabelList.append('decay'+'|'+self.materialsDict['MPI-'+str(mpi)][materialCounter]+'|'+self.isotopeList[i])
                  decayList.append(decayLine[i])
          if breakFlag == 1: break
    return decayLabelList, decayList
      
  def getDecayHeatMrtau(self,timeStepIndex,matchedTimeSteps):
    """
      Gets the decay heat from the standalone mrtau output
      @ In, timeStepIndex, integer, timestep number 
      @ In, matchedTimeSteps, list, list of time steps considered
      @ Out, decayListMrtau, list, list of the decay values from mrtau
    """
    decayFlag = 0
    breakFlag = 0
    self.decayLabelListMrtau = []
    self.numDensityLabelListMrtau = []
    decayListMrtau = []
    with open(os.path.join(self.workingDir,self.mrtauCSVOutputMPI), 'r') as outfile:  ### to fix: self.mrtauCSVOutputMPI is a list, so to open it has to loop through the mpi to open individually each file
      for line in outfile:
        if re.search(r'INDIVIDUAL DECAY HEAT BLOCK',line):
          decayFlag = 1
        if re.search(r'CUMULATIVE DECAY HEAT BLOCK',line):
          breakFlag = 1
        if decayFlag == 1 and breakFlag == 0:
          line = re.sub(r'\n',r'',line)
          decayLine = filter(None, line.split(' ')) 
          if decayLine != []: stringIsFloatNumber = self.isFloatNumber(decayLine) 
          if stringIsFloatNumber is True and decayLine != []:
            if (float(decayLine[0]) == float(matchedTimeSteps[timeStepIndex])):
              for i in xrange (0,len(self.isotopeListMrtau)):
                if int(self.perturbationNumber) == 1:
                  self.numDensityLabelListMrtau.append('numDensity'+'|'+self.isotopeListMrtau[i]) 
                  self.decayLabelListMrtau.append('decay'+'|'+self.isotopeListMrtau[i])
                decayListMrtau.append(decayLine[i+1])
          if breakFlag == 1:
            break 
    return decayListMrtau
  
  def getRRlist(self,instantDict):
    """
      Puts the RR labels and the RR values in a list 
      @ In, instantDict, dictionary   
      @ Out, RRnames, list, list of parameter labels that are in the RR matrix 
      @ Out, RRvalues, list, values relative to the labels, within the RR matrix
    """
    RRnames = []
    RRvalues = [] 
    if self.printSpatialRR is True:
      for i in xrange(0,len(self.paramList)):
        for j in xrange(1,int(self.Ngroups) + 1):
          for k in xrange(1, int(self.Nregions) + 1):
            if self.paramList[i] == 'Group': 
              pass
            else:
              RRnames.append(self.paramList[i]+'|gr'+str(j)+'|reg'+str(k))
              RRvalues.append(instantDict.get('reactionRateInfo').get(str(j)).get(str(k)).get(self.paramList[i]))
    if self.printSpatialRR is False:  
      for i in xrange(0,len(self.paramList)):
        if self.paramList[i]   == 'Group': 
          pass
        elif self.paramList[i] == 'Region': 
          pass
        else:
          RRnames.append(self.paramList[i]+'|Total')
          RRvalues.append(instantDict.get('reactionRateInfo').get(self.paramList[i]))
    if 'Group' in RRnames: 
      RRnames.remove('Group')
    return RRnames, RRvalues    
    
  def writeCSV(self,instantDict,timeStepIndex,matchedTimeSteps,jobTitle):
    """
      Prints the instant coupled to mrtau data in csv files 
      @ In, InstantDict, dictionary, contains all the values collected from instant output 
      @ In, timeStepIndex, integer, timestep number 
      @ In, matchedTimeSteps, list, list of time steps considered
      @ In, jobTitle, string, job title parsed from instant input 
      @ Out, None 
      
    """
    if self.paramList != []:
      RRnames, RRvalues = self.getRRlist(instantDict) 
      csvOutput = os.path.join(instantDict.get('workingDir'),jobTitle+'-'+self.perturbationNumber+'.csv')
      if self.phisicsRelap is False:
        with open(csvOutput, 'a+') as f:
          instantWriter = csv.writer(f, delimiter=',',quotechar=',', quoting=csv.QUOTE_MINIMAL)
          if timeStepIndex == 0:
            instantWriter.writerow(['timeMrTau'] + ['keff'] + ['errorKeff'] + RRnames + instantDict.get('fluxLabelList') + instantDict.get('matFluxLabelList') + instantDict.get('depLabelList') + instantDict.get('decayLabelList') + instantDict.get('XSlabelList') + ['cpuTime']) 
          instantWriter.writerow([str(matchedTimeSteps[timeStepIndex])] + instantDict.get('keff') + instantDict.get('errorKeff') + RRvalues + instantDict.get('fluxList') + instantDict.get('matFluxList') + instantDict.get('depList') + instantDict.get('decayList') + instantDict.get('XSlist') + instantDict.get('cpuTime'))
      if self.phisicsRelap is True: 
        with open(csvOutput, 'a+') as f:
          instantWriter = csv.writer(f, delimiter=',',quotechar=',', quoting=csv.QUOTE_MINIMAL)
          if timeStepIndex == 0:
            instantWriter.writerow(['timeMrTau'] + ['keff'] + ['errorKeff'] + RRnames + instantDict.get('fluxLabelList') + instantDict.get('powerDensLabelList') + instantDict.get('depLabelList') + instantDict.get('decayLabelList') + instantDict.get('XSlabelList') + ['cpuTime']) 
          instantWriter.writerow([str(matchedTimeSteps[timeStepIndex])] + instantDict.get('keff') + instantDict.get('errorKeff') + RRvalues + instantDict.get('fluxList') + instantDict.get('powerDensList') + instantDict.get('depList') + instantDict.get('decayList') + instantDict.get('XSlist') + [instantDict.get('cpuTime')])
      
  def writeMrtauCSV(self, mrtauDict):
    """
      Prints the mrtau standalone data in a csv file  
      @ In, mrtauDict, dictionary, contains all the values collected from instant output  
      @ Out, None 
    """ 
    csvOutput = os.path.join(mrtauDict.get('workingDir'),'mrtau'+'-'+self.perturbationNumber+'.csv')
    with open(csvOutput, 'a+') as f:
      mrtauWriter = csv.writer(f, delimiter=',',quotechar=',', quoting=csv.QUOTE_MINIMAL)
      if mrtauDict.get('timeStepIndex') == 0:
        mrtauWriter.writerow(['timeMrTau'] + self.numDensityLabelListMrtau + self.decayLabelListMrtau) 
      mrtauWriter.writerow( [str(mrtauDict.get('matchedTimeSteps')[mrtauDict.get('timeStepIndex')])] + mrtauDict.get('depList') + mrtauDict.get('decayHeatMrtau'))
    