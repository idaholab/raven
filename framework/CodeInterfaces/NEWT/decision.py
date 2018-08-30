"""
Created on Jan 18th, 2018
@author: rouxpn
""" 
import os
import re
import numpy as np
from decimal import Decimal
import matplotlib 
from collections import defaultdict
from collections import OrderedDict
import math 
from matplotlib.ticker import FormatStrFormatter
from matplotlib import collections as mc
matplotlib.use('Agg')
import pylab as plt
import random
import heapq

def __residuumSign(self):
  """
    calculates the residuum based on the phisics reference 
    @ In, None 
    @ Out, None 
  """
  refDict = {}
  parameters = []
  values = [] 
  count = 0 
  energy = [self.G1,self.G2,self.G3,self.G4,self.G5,self.G6]
  flux = [self.flux1,self.flux2,self.flux3,self.flux4,self.flux5,self.flux6]
  #energy = ['2.0000E+07','4.8000E+06','1.1300E+02','4.4000E+01','3.5000E+00','1.3000E+00']
  #flux = [444,777,888,999,66666,22222]
  baseFile = 'HTGR_test'
  with open(baseFile+'.csv', 'r') as outfile:
    for line in outfile:
      count = count + 1
      for i in xrange (0,len(line.split(','))):
        if count == 1:
          parameters.append(line.split(',')[i])
        if count == 2:
          values.append(line.split(',')[i]) 
  for i in xrange (0,len(parameters)):
    refDict[parameters[i]] = values[i]
  
  parsedFlux = ParseFlux(energy,flux)
  normFluxPerLetFineColl = parsedFlux.normFluxPerLethColl
  normFluxBroad = parsedFlux.normFlux
  normFluxPerLetBroad = parsedFlux.normFluxPerLethargyBroad
  randomNumCase = parsedFlux.randNum
  normFluxColl = parsedFlux.normFluxColl
  
  
  #upperBand = [upperEl * 1.50 for upperEl in normFluxPerLetFineColl]  # use of the normalized neutorn flux per unit lethergy
  #lowerBand = [lowerEl * 0.5 for lowerEl in normFluxPerLetFineColl]
  upperBand = [upperEl * 1.12 if normFluxColl > 0.5 else upperEl * 1.25 for upperEl in normFluxColl]   # use of the normalized neutron flux 
  lowerBand = [lowerEl * 0.88 if normFluxColl > 0.5 else upperEl * 0.75 for lowerEl in normFluxColl]
  
  keffDiff = (self.keff - float(refDict['keff']))*1e5
  # find the position of the two maximal ref flux values
  largest = 0
  secondLargest = 0
  for item in normFluxColl:
    if item > largest:
      largest = item
  for item in normFluxColl:    
    if largest > item > secondLargest:
      secondLargest = item
  positionOfMax = []
  for i in range (len(normFluxColl)):
    if normFluxColl[i] == largest or normFluxColl[i] == secondLargest:
      positionOfMax.append(i)  
  print "\n"
  print keffDiff
  if keffDiff > -200.0 and keffDiff < 200.0:
    passList = [1 if normFluxBroad[el] > lowerBand[el] and normFluxBroad[el] < upperBand[el] else 0 for el in range(len(normFluxBroad))]
    passList2 = [1 if normFluxBroad[el] > lowerBand[el] and normFluxBroad[el] < upperBand[el] else 0 for el in positionOfMax]
    ratioList = [normFluxColl[el] / normFluxBroad[el] for el in range(len(normFluxBroad))]
    screenedRatioList = [normFluxColl[el] / normFluxBroad[el] if normFluxColl[el] > 0.1 else 1.0 for el in range(len(normFluxBroad))]
    exceededRatio = [1 if el > 2.0 or el < 0.2 else 0 for el in screenedRatioList] # if a ratio is larger than 2.0 or smaller than 0.2, list element is 1, 0 otherwise. This is a failing condition regarless of the rest 
    print 'pass list 1'
    print passList
    print 'pass list 2'
    print passList2
    print ratioList
    print 'exceeded ratio'
    print exceededRatio
    print randomNumCase
    if sum(passList) >=  len(normFluxBroad) / 2 and sum(exceededRatio) == 0:
      print 'WAY 1 SUCCESS'
      return 1
    elif   sum(passList2) ==  len(positionOfMax) and sum(exceededRatio) == 0:
      print 'WAY 2 SUCCESS'
      return 1 
    else: 
      return -1
  else: 
    return -1

class ParseFlux:   
  """
    Parses the PHISICS output to get the neutron flux within each group, and plot it. 
    required input:
    - reference phisics output in 252 groups (name: HTGR-test.o0)
    - sampled phisics output in collasped groups (in ./adapt/P/HTGR-test.oN) where P is the sample number, and N is the node numbder (in MPI)
    - reference newt calculation in 249 (located in the root directory ./newt0.out)
    - newt calculation to get the collpased structure of each perturbation (located in the subdirectory ./adapt/P/newt0.out)
  """
  def __init__(self,energy,flux):
    """
      Distribute te various functions 
      @ In, baseFile, string, base name (no extension) of the phisics output file to be parsed
      @ Out, None 
    """
    mpi = 0
    reactDict252 = lambda: defaultdict(reactDict252)
    myReactDict252 = reactDict252()
    resultDict = {}
    resultDict['group'] = {}
    resultDict['total'] = {}
    resultDict['group249'] = {}
    resultDict['total249'] = {}
    resultDict['collapsedStruct'] = {}
    resultDict['upperBounds249'] = {}
    self.workingDir = os.getcwd()
    output = os.path.join(self.workingDir,'HTGR-test.o0') # the ref 252 is located in the root directory
    self.getNumberOfRegions(output)
    groupRR = self.getReactionRates(output,mpi,myReactDict252)
    resultDict['group249']['pert1'] = self.summedDictValues('group',groupRR)
    resultDict['upperBounds249']['pert1'] = self.getCollapsedStruct(os.path.join(self.workingDir,'newt0.out'))
    energy249Dupli, flux249Dupli, fission249Dupli, normFlux249Dupli, energy249, flux249 = self.plot(resultDict,'group249','upperBounds249','1')
    normFluxPerLethargy249 = self.getFluxPerLethargy(energy249,flux249)
    self.normFluxColl = self.collFlux(energy249,energy,flux249)
    normFluxCollDupli = self.duplicateValues(self.normFluxColl,'y')
    self.normFluxPerLethColl = self.collapseFine(energy249,energy,normFluxPerLethargy249)
    normFluxPerLethCollDupli = self.duplicateValues(self.normFluxPerLethColl,'y')
    energyDupli = self.duplicateValues(energy,'x')
    self.normFlux = self.normalize(energy,flux)
    normFluxDupli = self.duplicateValues(self.normFlux,'y')
    
    self.normFluxPerLethargyBroad = self.getFluxPerLethargy(energy,flux)
    normFluxPerLethargyBroadDupli = self.duplicateValues(self.normFluxPerLethargyBroad,'y')
      
      
    # call the figure plotting 
    pert = 1 
    #self.figure(energyDupli, normFluxDupli,'NormFluxPerLetColl','Energy','Normalized Flux per unit lethargy',energyDupli,normFluxPerLethCollDupli,'pert'+str(pert),'norm')
    #self.figure(energyDupli, normFluxPerLethargyBroadDupli,'NormLet','Energy','Normalized Flux per unit lethargy',energyDupli,normFluxPerLethCollDupli,'pert'+str(pert),'norm')
    self.figure(energyDupli, normFluxDupli,'NormFlux','Energy','Normalized Flux',energyDupli,normFluxCollDupli,'pert'+str(pert),'norm')
    
  def getNumberOfRegions(self,output):
    """
      Gives the number of spatial regions used in the PHISICS simulations.
      @ In, output, string, PHISICS output file.
      @ Out, None
    """
    flagStart = 0
    count = 0
    subdomainList = []
    with open(output, 'r') as outfile:
      for line in outfile:
        if re.search(r'Subdomain volumes',line):
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
    self.nRegions = count
    return
    
  def isNumber(self,line):
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
      except ValueError: # the instance is a string, for example
        return False

  def getReactionRates(self,output,mpi,myReactDict):
    """
      Gets the reactions rates, power for each group in PHISICS
      @ In, output, string, PHISICS output file. 
      @ In, mpi, integer, node considered
      @ In, myReactDict, dictionary, reaction rate dictrionary
      @ Out, myReactDict, dictionary, dictionary containing the reaction rate info
    """
    flagStart = 0
    count = 0

    with open(output, 'r') as outfile:
      for line in outfile:
        if re.search(r'averaged flux\s+power', line): # beginning of the reaction rate matrix 
          numberingRR = self.mapColumns(line,count)
          self.paramList = []
          self.paramList = numberingRR.keys()
          flagStart = 1
        if re.search(r'Fission matrices of all',line): # end of the reaction rate matrix 
          flagStart = 2
        if flagStart == 1: # the parsing starts here. It ends as soon as flagStart = 2  
          if re.search(r'\w+\s+\w+',line):
            line = re.sub(r'[\|]',' ',line)
            lineSplit = line.split()
            stringIsNumber = self.isNumber(lineSplit)
            if stringIsNumber :
              for reactRate in numberingRR:
                groupNum  = lineSplit[0]
                regionNum = lineSplit[1]
                if reactRate != 'Group' and reactRate != 'Region':
                  myReactDict[groupNum][regionNum][reactRate][mpi] = lineSplit[numberingRR.get(reactRate)]
      return myReactDict

  def summedDictValues(self,groupOrTot, nestedDict):
    """
      Sums the values from the deepest nest of a dictionary. The values summed are the MPI values, for each parameter.
      @ In, nestedDict, dictionary, nested dictionaries of intergers or floats
      @ In, mpiOrTot, string, equal to 'group' or 'total'. indicates if group RR or total RR are passed to this function
      @ Out, summedDict, dictionary, dictionary of integer or float
    """
    summedDict = lambda: defaultdict(summedDict)
    mySummedDict = summedDict()
    if groupOrTot == 'total':
      for parameter,mpi in nestedDict.iteritems():
        sumString =  '%.15E' % Decimal(  sum([float(elm) for elm in mpi.values()]) )
        mySummedDict[parameter] = sumString
    if groupOrTot == 'group':
      for group,region in nestedDict.iteritems():
        for region,parameter in region.iteritems():
          for parameter,mpi in parameter.iteritems():
            sumString =  '%.15E' % Decimal(  sum([float(elm) for elm in mpi.values()]) )
            mySummedDict[group][region][parameter] = sumString
    return mySummedDict      

  def mapColumns(self,line,count):
    """
      Allocates a column number relative to the reaction rates.
      @ In, line, string
      @ In, count, interger, counts the column position
      @ Out, numbering, dictionary, key: reaction rate name, value: column number
    """
    numbering = {}
    line = re.sub(r'averaged',r'',line)
    line = re.sub(r'fis. ',r'',line)
    line = re.sub(r'[\|]',' ',line)
    parameterNames = line.split()
    for param in parameterNames:
      numbering[param] = count
      count +=  1
    return numbering
   
  def getCollapsedStruct(self, newtInput):
    """
      Get the structure used in the newt input
      @ In, newtInput, string, path to the newt input
      @ out, upperBoundary, list, list of energy upper boundaries 
    """
    upperBoundary = []
    startFlag = 0
    endFlag = 0
    homogenizedMixtures = []
    with open(newtInput, 'r') as infile: 
      for line in infile:
        if re.search(r'Broad Group Parameters',line.strip()):
          startFlag = 1 
        if startFlag == 1 and endFlag == 0:
          lineSplit = filter(None, line.split(' '))
          stringIsNumber = self.isNumber(lineSplit)
          if stringIsNumber:
            upperBoundary.append(lineSplit[1]) 
        if re.search(r'Cell Averaged Fluxes',line.strip()) and startFlag == 1:
          endFlag = 1
          break
    return upperBoundary 

  def getFluxPerLethargy(self,energy,flux):
    """
      calculates the flux per unit lethargy
      @ In, energy, list, energy boundaries
      @ In, flux, list, flux values 
      @ NormFluxPerLeth, list, normalized neutron flux per unit lerthargy
    """
    fluxPerLethargy = []
    NormFluxPerLethargy = []
    for e in range (len(flux)):
      if e == len(flux) - 1:
        fluxPerLethargy.append(float(flux[e]) / (math.log(float(energy[e])) - math.log(1.0E-5))) 
      else:
        fluxPerLethargy.append(float(flux[e]) / (math.log(float(energy[e])) - math.log(float(energy[e+1]))))
    for e in range (len(flux)):
        NormFluxPerLethargy.append(fluxPerLethargy[e] / sum(fluxPerLethargy)) 
    return NormFluxPerLethargy  
  
  def collFlux(self,struct249,structBroad,funcToColl):
    """
      Integrates (i.e. sums) the fine group data into the broad group data of interest.  
      @ In, struct249, list, energy group structure in 249 groups
      @ In, structBroad, list, broad energy group structure
      @ In, funcToColl, list, function to be collasped from 249 to broad gorup struct
      @ Out, funcCollapsed
    """
    totalFunc = 0
    count = 0     
    energyIndex = []
    sumValues = 0 
    collapsedSum = []
    normCollapsed = []
    individualSum = []
    # make sure the broad structure is in scientific notation
    structBroad = self.scientificNotation(structBroad)
    for eBroad in structBroad:
      energyIndex.append(struct249.index(eBroad))
    for eFine in range (len(funcToColl)):
      f = float(funcToColl[eFine])
      if eFine == len(funcToColl) - 1:
        highE = float(struct249[eFine])
        lowE =  0.000
      else: 
        highE = float(struct249[eFine])
        lowE =  float(struct249[eFine + 1])  
      individualSum.append(f * (highE - lowE))
    #individualSum = [float(funcToColl[eFine]) * (float(struct249[eFine]) - 0.000) if eFine == len(funcToColl) else float(funcToColl[eFine]) * (float(struct249[eFine-1]) - float(struct249[eFine])) for eFine in range (len(funcToColl))] 
    for eFine in range (len(individualSum)):
      sumValues = sumValues + float(individualSum[eFine])
      if eFine + 1 in list(set(energyIndex)): # if the index of the fine structure corresponds to one of the broad boundaries
        count = count + 1
        if eFine == 0:
          if len(structBroad) == len(struct249):  # if the structure is collaspe group per group (remain in fine group struct)
            sumValues = sumValues + float(individualSum[eFine])
            collapsedSum.append(sumValues/(float(structBroad[count-1])-float(structBroad[count])))
          # The first energy boundary is in common with the braod and fine group, but this only the first value of the flux. The values needs to be summed until the next boundary.
          pass    
        else:
          collapsedSum.append(sumValues/(float(structBroad[count-1])-float(structBroad[count])))
          sumValues = 0
      if eFine + 1 == len(funcToColl) - 1 :# corresponds to the last broad group (from the most thermal upper boundary of ther broad structure to the energy e = 1.00E-5 eV) 
        collapsedSum.append(sumValues/(float(structBroad[count])-0.000))
        sumValues = 0
    for collSum in collapsedSum:
      normCollapsed.append(collSum / sum(collapsedSum)) 
    return normCollapsed
  
  def collapseFine(self,struct249,structBroad,funcToColl):
    """
      Integrates (i.e. sums) the fine group data into the broad group data of interest.  
      @ In, struct249, list, energy group structure in 249 groups
      @ In, structBroad, list, broad energy group structure
      @ In, funcToColl, list, function to be collasped from 249 to broad gorup struct
      @ Out, funcCollapsed
    """
    totalFunc = 0 
    energyIndex = []
    sumValues = 0 
    collapsedSum = []
    normCollapsed = []
    # make sure the broad structure is in scientific notation
    structBroad = self.scientificNotation(structBroad)
    for eBroad in structBroad:
      energyIndex.append(struct249.index(eBroad))
    for eFine in range (len(funcToColl)):
      sumValues = sumValues + float(funcToColl[eFine])
      if eFine + 1 in list(set(energyIndex)): # if the index of the fine structure corresponds to one of the broad boundaries
        if eFine == 0:
          # The first energy boundary is in common with the braod and fine group, but this only the first value of the flux. The values needs to be summed until the next boundary.
          pass    
        else:
          collapsedSum.append(sumValues)
          sumValues = 0
      if eFine + 1 == len(funcToColl) - 1 :# corresponds to the last broad group (from the most thermal upper boundary of ther broad structure to the energy e = 1.00E-5 eV) 
        collapsedSum.append(sumValues)
        sumValues = 0
    for collSum in collapsedSum:
      normCollapsed.append(collSum / sum(collapsedSum)) 
    return normCollapsed
    
  def scientificNotation(self,givenList):
    """
      Converts the numerical values into a scientific notation.
      @ In, givenList, lsit, list of values potentially not in scientific notation
      @ Out, scientificList, lsit, list of values in scientific notation
    """
    scientificList = []
    for el in givenList:
      scientificList.append('%.4E' % Decimal(str(el))) 
    return scientificList
    
  def figure(self,x,y,name,xLabel,yLabel,xRef,yRef,pert,scale):
    """
      figure template
      @ In, x, list, axis values
      @ In, y, list, ordinate values
      @ In, name, string, file name of the plot
      @ In, xlabel, string, x axis label 
      @ In, ylabel, string, y axis label
      @ In, xref, string, ref plot on x-axis 
      @ In, yref, string, ref plot on y-axis
      @ In, pert, string, perturbation number (used in the name)
      @ In, scale, string, 'log', 'norm' 'none', for the scale y-axis
      @ Out, None  
    """
    self.randNum = str(random.randint(1,1000000000000))
    #stddev = self.statistics(y)
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    #ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xticks([1e-4,1e-3,1e-2,1e-1,1, 10, 100,1000,1e4,1e5,1e6,1e7])
      
    ax = plt.gca()
    plt.plot(x,y,'r-',label='Broad grp flux')
    plt.plot(xRef,yRef,'b-',label='Ref flux')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend(loc='upper left')
    ax.axis('auto')
    if scale == 'norm':
      ax.set_xlim([1.0e-5,1e7])
      ax.set_ylim([0.0,1.0])
    #matplotlib.pyplot.xticks(x)
    #plt.text(0.70, 0.95, r'stddev = '+str(stddev)+'%',transform=ax.transAxes)
    plt.subplots_adjust(left=0.15)
    plt.savefig(name+self.randNum+'.png')
    
    matplotlib.pyplot.close()    

  def plot(self,resultDict,nGroupsKey,bounds,pert):
    """
      plot the flux as a function of the energy for the reference . 
      @ In, resultDict, dictionarry of power values
      @ In, nGroupsKey, string, 'group249' if reference is considered, 'group' otherwise
      @ In, bounds, string, key word relative to the upper boundaries to be user: upperBounds249 or collapsedStruct
      @ In, pert, interger or string, indicates the perturbation number, is equ
      @ Out, energyBroadDupli, list, list of energy boundaries duplicated 
      @ Out, fluxDupli, list, list of flux duplicated 
      @ Out, fissionDupli, list, list of fission rates duplicated 
    """
    workingDict = resultDict.get(nGroupsKey).get('pert'+str(pert))
    fissionDict = {}
    fluxDict = {}
    groupOrdered = []
    groupNum = []
    fluxOrdered = []
    fissionOrdered = []
    regionList = []
    for group,region in workingDict.iteritems():
      for region,variable in region.iteritems():
        regionList.append(int(region))
        for variable,value in variable.iteritems():
          # select the region and reaction you want to plot
          if region == '1' and variable == 'flux':
            groupNum.append(int(group)) 
            fluxDict[group] = value
          if region == '1' and variable == 'fission':
            fissionDict[group] = value
    for i in range (len(groupNum)):
      groupOrdered.append(i+1)  # group numbers, ordered
      fluxOrdered.append(fluxDict.get(str(i+1))) # flux values, ordered
      fissionOrdered.append(fissionDict.get(str(i+1))) # flux values, ordered
    energy = self.equivalenceGroupToEnergy(groupOrdered,resultDict.get(bounds).get('pert'+str(pert)))
    normFluxOrdered = self.normalize(energy,fluxOrdered)
    self.normFlux249 = normFluxOrdered 
    energyDupli = self.duplicateValues(energy,'x')
    fluxDupli = self.duplicateValues(fluxOrdered,'y')
    fissionDupli = self.duplicateValues(fissionOrdered,'y')
    normFluxDupli = self.duplicateValues(normFluxOrdered,'y')
    return energyDupli, fluxDupli, fissionDupli, normFluxDupli, energy, fluxOrdered
  
  def equivalenceGroupToEnergy(self,group,upperBounds):
    """
    The group numbers extracted in the list group252 are unordered (they are not going from 1 to 249 increasingly). 
    This method reads the unordered list of groups, locates the energy (in eV) it corresponds to, and put the energy 
    value in a list. 
    @ In, group249, list, list of unordered groups
    @ In, upperBounds, list, upper boundary energies, ordered from high energies to low energies
    @ Out, energy249, list, unordered energies
    """
    energy = []
    for e in range (len(group)):
      energy.append(upperBounds[e])
    return energy
 
  def normalize(self,energy,funcToNormalize):
    """
      Normalizes the flux, fission rate etc.. 
      @ In, energy, list, list of energy boundaries
      @ In, funcToNormalize, list, function to normalize (i.e. flux, fission rate etc..)
      @ Out, normalizedFunc, list, list of values normalized
    """
    totalF = 0
    normalizedFunc = []
    eRange = []
    for e in range(len(energy)):
      if e < len(energy) - 1: # all the energy values expect the last one
        eRange.append(float(energy[e]) - float(energy[e+1]))
      else: 
        eRange.append(float(energy[e]) - 0.0) # the energy 0.0 eV is added at the end to calculate the last energy range
      #productEnergy = eRange[e] * float(funcToNormalize[e])
      productEnergy =  float(funcToNormalize[e])
      totalF = totalF + productEnergy
    for e in range(len(energy)):
      #normalizedFunc.append(float(funcToNormalize[e]) * eRange[e] / totalF)
      normalizedFunc.append(float(funcToNormalize[e]) / totalF)
    return normalizedFunc
      
      
  def duplicateValues(self,inList,axis):
    """
      Duplicate the values in a list in order to have a continuous-by-piece function. 
      Example: if a list has the values [1, 2 , 3, 4], it will be turned in 
      [1, 2, 2, 3, 3, 4]. 
      @ In, inList, list, list of values to be duplicated
      @ In, axis, string, specifies if this is for x-axis values or y-axis values. The x and y axis values of a same graph are staggered, and hence need to be treated differently
      @ Out, outList, list, duplicated list
    """
    outList = []
    for i in range (len(inList)):
      if i == 0 and axis == 'x':
        # highest fast energy exists only once
        outList.append(inList[i])
      elif i == len(inList) - 1 and axis == 'x':
        # lowest thermal energy exists only once
        outList.extend((inList[i],inList[i]))
        outList.append(1.00E-05)
      else:
        outList.extend((inList[i],inList[i]))
    return outList    
    
    
#f= __residuumSign(1)   
    