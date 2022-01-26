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
Created on Nov 14, 2013

@author: alfoa
"""
import numpy as np
import ast
import copy
import numpy.ma as ma
import os
import re
import gc
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

from utils.InputData import parameterInputFactory as PIF
from utils import utils, mathUtils
from utils import mathUtils
from .PlotInterface import PlotInterface
from ClassProperty import ClassProperty

import matplotlib.pyplot as plt

display = utils.displayAvailable()

class GeneralPlot(PlotInterface):
  """
    OutStream of type Plot
  """

  ## Promoting these to static class variables, since they will not alter from
  ## object to object. The use of the @ClassProperty with only a getter makes
  ## the variables immutable (so long as no one touches the internally stored
  ## "_"-prefixed), so other objects don't accidentally modify them.

  # TODO these should be moved into the InputParams
  ## available 2D and 3D plot types
  _availableOutStreamTypes = {2:['scatter', 'line', 'histogram', 'stem', 'step',
                                 'pseudocolor', 'dataMining', 'contour',
                                 'filledContour'],
                              3:['scatter', 'line', 'histogram', 'stem',
                                 'surface', 'wireframe', 'tri-surface',
                                 'contour', 'filledContour']}

  ## interpolate functions available
  _availableInterpolators = ['nearest', 'linear', 'cubic', 'multiquadric',
                             'inverse', 'gaussian', 'Rbflinear', 'Rbfcubic',
                             'quintic', 'thin_plate']

  @ClassProperty
  def availableOutStreamTypes(cls):
    """
        A class level constant that tells developers what outstreams are
        available from this class
        @ In, cls, the OutStreamPlot class of which this object will be a type
    """
    return cls._availableOutStreamTypes

  @ClassProperty
  def availableInterpolators(cls):
    """
        A class level constant that tells developers what interpolators are
        available from this class
        @ In, cls, the OutStreamPlot class of which this object will be a type
    """
    return cls._availableInterpolators

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class "cls".
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying the input of cls.
    """
    spec = super().getInputSpecification()
    # TODO this is waaaaay to much to convert right now
    # For now, accept a blank plotting check and sort it out later.
    spec.strictMode = False
    return spec
    ###################################################################
    # TODO here's a good start, but skipping for now:
    # spec.addParam('interactive', param_type=InputTypes.BoolType)
    # spec.addParam('overwrite', param_type=InputTypes.BoolType)
    # spec.addSub(PIF('filename', contentType=InputTypes.StringType))

    # xyz = InputTypes.makeEnumType('PlotXYZ', 'PlotXYZ', ['x', 'y', 'z'])

    # action = PIF('actions')
    # hows = InputTypes.makeEnumType('GeneralPlotHow', 'GeneralPlotHow',
    #        ['screen', 'pdf', 'png', 'eps', 'pgf', 'ps', 'gif', 'svg', 'jpeg', 'raw', 'bmp', 'tiff', 'svgz'])
    # action.addSub(PIF('how', contentType=hows))

    # title = PIF('title')
    # title.addSub(PIF('text', contentType=InputTypes.StringType))
    # # kwargs can be anything, so just turn strict mode off for it
    # title.addSub(PIF('kwargs', strictMode=False))
    # action.addSub(title)

    # labelFormat = PIF('labelFormat')
    # labelFormat.addSub(PIF('axis', contentType=xyz))
    # sciPlain = InputTypes.makeEnumType('SciNot', 'SciNot', ['sci', 'scientific', 'plain'])
    # labelFormat.addSub(PIF('style', contentType=sciPlain))
    # labelFormat.addSub(PIF('scilimits', contentType=InputTypes.StringType))
    # labelFormat.addSub(PIF('useOffset', contentType=InputTypes.FloatType))
    # action.addSub(labelFormat)

    # figProp TODO WORKING XXX
    # TODO
    # spec.addSub(action)

    # settings = parameterInputFactory('plotSettings')
    # TODO
    # spec.addSub(settings)
    # return spec
    #################################### END draft

  def __init__(self):
    """
      Initialization method defines the available plot types, the identifier of
      this object and sets default values for required data.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'OUTSTREAM PLOT'
    self.options = {}    # outstreaming options # no addl info from original developer
    self.counter = 0     # counter # no additional info given by original developer

    ## default plot is 2D
    self.dim = None

    ## list of source names
    self.sourceName = []

    ## source of data
    self.sourceData = None

    ## x,y,z coordinate names
    self.xCoordinates = None
    self.yCoordinates = None
    self.zCoordinates = None

    ## dictionary of x,y,z values
    self.xValues = None
    self.yValues = None
    self.zValues = None

    ## color map
    self.colorMapCoordinates = {}
    self.colorMapValues = {}

    ## list of the outstream types
    self.outStreamTypes = []

    ## actual plot (MPL axes object)
    self.actPlot = None
    self.gridSpace = None

    ## MPL colormap object
    self.actcm = None

    ## For the data-mining plot, I think?
    self.clusterLabels = None
    self.clusterValues = None

    ## Gaussian Mixtures
    self.mixtureLabels = None
    self.mixtureValues = None
    self.mixtureMeans = None
    self.mixtureCovars = None

  # TODO started, but didn't finish due to time constraints
  # this should be a good start for _handleInput in the future.
  # def _handleInput(self, spec):
  #   """
  #     Loads the input specs for this object.
  #     @ In, spec, InputData.ParameterInput, input specifications
  #     @ Out, None
  #   """
  #   if 'dim' in spec.parameterValues:
  #     self.raiseAnError(IOError,"the 'dim' attribute has been deprecated. This warning became an error in January 2017")
  #   foundPlot = False
  #   for subnode in spec.subparts:
  #     # if actions, read actions block
  #     if subnode.getName() == 'filename':
  #       self.filename = subnode.value
  #     if subnode.getName() == 'actions':
  #       self.__readPlotActions(subnode)
  #     if subnode.getName() == 'plotSettings':
  #       self.options[subnode.getName()] = {}
  #       self.options[subnode.getName()]['plot'] = []
  #       for subsub in subnode.subparts:
  #         if subsub.getName() == 'gridSpace':
  #           # if self.dim == 3: self.raiseAnError(IOError, 'SubPlot option can not be used with 3-dimensional plots!')
  #           self.options[subnode.getName()][subsub.getName()] = subsub.value
  #         elif subsub.getName() == 'plot':
  #           tempDict = {}
  #           foundPlot = True
  #           for subsubsub in subsub.subparts:
  #             if subsubsub.getName() == 'gridLocation':
  #               tempDict[subsubsub.getName()] = {}
  #               for subsubsubsub in subsubsub.subparts:
  #                 tempDict[subsubsub.getName()][subsubsubsub.getName()] = subsubsubsub.value
  #             elif subsubsub.getName() == 'range':
  #               tempDict[subsubsub.getName()] = {}
  #               for subsubsubsub in subsubsub.subparts:
  #                 tempDict[subsubsub.getName()][subsubsubsub.getName()] = subsubsubsub.value
  #             elif subsubsub.getName() != 'kwargs':
  #               tempDict[subsubsub.getName()] = subsubsub.value
  #             else:
  #               tempDict['attributes'] = {}
  #               for sss in subsubsub.subparts:
  #                 tempDict['attributes'][sss.getName()] = sss.value
  #           self.options[subnode.getName()][subsub.getName()].append(tempDict)
  #         elif subsub.getName() == 'legend':
  #           self.options[subnode.getName()][subsub.getName()] = {}
  #           for legendChild in subsub.subparts:
  #             self.options[subnode.getName()][subsub.getName()][legendChild.getName()] = utils.tryParse(legendChild.value)
  #         else:
  #           self.options[subnode.getName()][subsub.getName()] = subsub.value
  #     # TODO WORKING XXX
  #     if subnode.getName() == 'title':
  #       self.options[subnode.getName()] = {}
  #       for subsub in subnode:
  #         self.options[subnode.getName()][subsub.getName()] = subsub.text.strip()
  #       if 'text'     not in self.options[subnode.getName()].keys():
  #         self.options[subnode.getName()]['text'    ] = xmlNode.attrib['name']
  #       if 'location' not in self.options[subnode.getName()].keys():
  #         self.options[subnode.getName()]['location'] = 'center'
  #     ## is this 'figureProperties' valid?
  #     if subnode.getName() == 'figureProperties':
  #       self.options[subnode.getName()] = {}
  #       for subsub in subnode:
  #         self.options[subnode.getName()][subsub.getName()] = subsub.text.strip()
  #   self.type = 'OutStreamPlot'

  #   if not 'plotSettings' in self.options.keys():
  #     self.raiseAnError(IOError, 'For plot named ' + self.name + ' the plotSettings block is required.')

  #   if not foundPlot:
  #     self.raiseAnError(IOError, 'For plot named' + self.name + ', No plot section has been found in the plotSettings block!')

  #   self.outStreamTypes = []
  #   xyz, xy             = sorted(['x','y','z']), sorted(['x','y'])
  #   for pltIndex in range(len(self.options['plotSettings']['plot'])):
  #     if not 'type' in self.options['plotSettings']['plot'][pltIndex].keys():
  #       self.raiseAnError(IOError, 'For plot named' + self.name + ', No plot type keyword has been found in the plotSettings/plot block!')
  #     else:
  #       # check the dimension and check the consistency
  #       if set(xyz) < set(self.options['plotSettings']['plot'][pltIndex].keys()):
  #         dim = 3
  #       elif set(xy) < set(self.options['plotSettings']['plot'][pltIndex].keys()):
  #         dim = 2 if self.options['plotSettings']['plot'][pltIndex]['type'] != 'histogram' else 3
  #       elif set(['x']) < set(self.options['plotSettings']['plot'][pltIndex].keys()) and self.options['plotSettings']['plot'][pltIndex]['type'] == 'histogram':
  #         dim = 2
  #       else:
  #         self.raiseAnError(IOError, 'Wrong dimensionality or axis specification for plot '+self.name+'.')
  #       if self.dim is not None and self.dim != dim:
  #         self.raiseAnError(IOError, 'The OutStream Plot '+self.name+' combines 2D and 3D plots. This is not supported!')
  #       self.dim = dim
  #       if self.availableOutStreamTypes[self.dim].count(self.options['plotSettings']['plot'][pltIndex]['type']) == 0:
  #         self.raiseAMessage('For plot named' + self.name + ', type ' + self.options['plotSettings']['plot'][pltIndex]['type'] + ' is not among pre-defined plots! \n The OutstreamSystem will try to construct a call on the fly!', 'ExceptedError')
  #       self.outStreamTypes.append(self.options['plotSettings']['plot'][pltIndex]['type'])
  #   self.raiseADebug('matplotlib version is ' + str(matplotlib.__version__))

  #   if self.dim not in [2, 3]:
  #     self.raiseAnError(TypeError, 'This Plot interface is able to handle 2D-3D plot only')

  #   if 'gridSpace' in self.options['plotSettings'].keys():
  #     grid = list(map(int, self.options['plotSettings']['gridSpace'].split(' ')))
  #     self.gridSpace = matplotlib.gridspec.GridSpec(grid[0], grid[1])


  #####################
  #  PRIVATE METHODS  #
  #####################

  def __splitVariableNames(self, what, where):
    """
      Function to split the variable names
      @ In, what, string,  x,y,z or colorMap
      @ In, where, tuple, where[0] = plotIndex, where[1] = variable Index
      @ Out, result, list, splitted variable
    """
    if   what == 'x':
      var = self.xCoordinates[where[0]][where[1]]
    elif what == 'y':
      var = self.yCoordinates[where[0]][where[1]]
    elif what == 'z':
      var = self.zCoordinates[where[0]][where[1]]
    elif what == 'colorMap':
      var = self.colorMapCoordinates[where[0]][where[1]]
    elif what == 'clusterLabels':
      var = self.clusterLabels[where[0]][where[1]]
    elif what == 'mixtureLabels':
      var = self.mixtureLabels[where[0]][where[1]]
    elif what == 'mixtureMeans':
      var = self.mixtureMeans[where[0]][where[1]]
    elif what == 'mixtureCovars':
      var = self.mixtureCovars[where[0]][where[1]]

    ## The variable can contain brackets {} (when the symbol "|" is present in
    ## the variable name), e.g.:
    ##        DataName|Input|{RavenAuxiliary|variableName|initial_value}
    ## or it can look like:
    ##                       DataName|Input|variableName

    if var != None:
      result = [None] * 3
      if   '|input|'  in var.lower():
        match = re.search(r"(\|input\|)", var.lower())
      elif '|output|' in var.lower():
        match = re.search(r"(\|output\|)", var.lower())
      else:
        self.raiseAnError(IOError, 'In Plot ' + self.name + ', the input coordinate ' + what + ' has not specified an "Input" or "Output" (case insensitive). e.g., sourceName|Input|aVariable) in ' + var)
      startLoc, endLoc = match.start(), match.end()
      result[0], result[1], result[2] = var[:startLoc].strip(), var[startLoc + 1:endLoc - 1].strip(), var[endLoc:].strip()
      if '{' in result[-1] and '}' in result[-1]:
        locLower, locUpper = result[-1].find("{"), result[-1].rfind("}")
        result[-1] = result[-1][locLower + 1:locUpper].strip()
    else:
      result = None
    return result

  def __readPlotActions(self, sNode):
    """
      Function to read, from the xml input, the actions that are required to be
      performed on this plot
      @ In, sNode, xml.etree.ElementTree.Element, xml node containing the action XML node
      @ Out, None
    """
    for node in sNode:
      self.options[node.tag] = {}
      if len(node):
        for subnode in node:
          if subnode.tag != 'kwargs':
            self.options[node.tag][subnode.tag] = subnode.text
            #if not subnode.text:
            #  self.raiseAnError(IOError, 'In Plot ' + self.name + '. Problem in sub-tag ' + subnode.tag + ' in ' + node.tag + ' block. Please check!')
          else:
            self.options[node.tag]['attributes'] = {}
            for subsub in subnode:
              try:
                self.options[node.tag]['attributes'][subsub.tag] = ast.literal_eval(subsub.text)
              except:
                self.options[node.tag]['attributes'][subsub.tag] = subsub.text
              if not subnode.text:
                self.raiseAnError(IOError, 'In Plot ' + self.name + '. Problem in sub-tag ' + subnode.tag + ' in ' + node.tag + ' block. Please check!')
      elif node.text:
        if node.text.strip():
          ## This is not great, we are over complicating this data structure if
          ## we have to double represent something in a dictionary... Some how
          ## this needs to be reworked
          self.options[node.tag][node.tag] = node.text
    ## There is something wrong here, why do we need to add an extra level of
    ## abstraction here? Why not self.options['how'] = 'screen'?
    if 'how' not in self.options.keys():
      self.options['how'] = {'how':'screen'}

  def __fillCoordinatesFromSource(self):
    """
      Function to retrieve the pointers of the data values (x,y,z)
      @ In, None
      @ Out, __fillCoordinatesFromSource, bool, true if the data are filled, false otherwise
    """
    self.xValues = []
    if self.yCoordinates:
      self.yValues = []
    if self.zCoordinates:
      self.zValues = []
    if self.clusterLabels:
      self.clusterValues = []
    if self.mixtureLabels:
      self.mixtureValues = []
    # if self.colorMapCoordinates[pltIndex] != None: self.colorMapValues = []
    for pltIndex in range(len(self.outStreamTypes)):
      self.xValues.append(defaultdict(list))
      if self.yCoordinates:
        self.yValues.append(defaultdict(list))
      if self.zCoordinates:
        self.zValues.append(defaultdict(list))
      if self.clusterLabels:
        self.clusterValues.append(defaultdict(list))
      if self.mixtureLabels:
        self.mixtureValues.append(defaultdict(list))
      if self.colorMapCoordinates[pltIndex] is not None:
        self.colorMapValues[pltIndex] = defaultdict(list)
    for pltIndex in range(len(self.outStreamTypes)):
      if len(self.sourceData[pltIndex]) == 0:
        return False
      dataSet = self.sourceData[pltIndex].asDataset()
      if self.sourceData[pltIndex].type.strip() != 'HistorySet':
        for i in range(len(self.xCoordinates [pltIndex])):
          xSplit = self.__splitVariableNames('x', (pltIndex, i))
          if xSplit[2].strip() not in self.sourceData[pltIndex].getVars(xSplit[1].lower()):
            self.raiseAnError(IOError, 'Not found variable "'+ xSplit[2] + '" in "'+xSplit[1]+ '" of DataObject "'+ self.sourceData[pltIndex].name+'"!')
          self.xValues[pltIndex][1].append(np.asarray(dataSet[xSplit[2]].values.astype(float, copy=False)))
        if self.yCoordinates :
          for i in range(len(self.yCoordinates [pltIndex])):
            ySplit = self.__splitVariableNames('y', (pltIndex, i))
            if ySplit[2].strip() not in self.sourceData[pltIndex].getVars(ySplit[1].lower()):
              self.raiseAnError(IOError, 'Not found variable "'+ ySplit[2] + '" in "'+ySplit[1]+ '" of DataObject "'+ self.sourceData[pltIndex].name+'"!')
            self.yValues[pltIndex][1].append(np.asarray(dataSet[ySplit[2].strip()].values.astype(float, copy=False)))
        if self.zCoordinates  and self.dim > 2:
          for i in range(len(self.zCoordinates [pltIndex])):
            zSplit = self.__splitVariableNames('z', (pltIndex, i))
            if zSplit[2].strip() not in self.sourceData[pltIndex].getVars(zSplit[1].lower()):
              self.raiseAnError(IOError, 'Not found variable "'+ zSplit[2] + '" in "'+zSplit[1]+ '" of DataObject "'+ self.sourceData[pltIndex].name+'"!')
            self.zValues[pltIndex][1].append(np.asarray(dataSet[zSplit[2].strip()].values.astype(float, copy=False)))
        if self.clusterLabels:
          for i in range(len(self.clusterLabels [pltIndex])):
            clusterSplit = self.__splitVariableNames('clusterLabels', (pltIndex, i))
            if clusterSplit[2].strip() not in self.sourceData[pltIndex].getVars(clusterSplit[1].lower()):
              self.raiseAnError(IOError, 'Not found variable "'+ clusterSplit[2] + '" in "'+clusterSplit[1]+ '" of DataObject "'+ self.sourceData[pltIndex].name+'"!')
            self.clusterValues[pltIndex][1].append(np.asarray(dataSet[clusterSplit[2].strip()].values.astype(float, copy=False)))
        if self.mixtureLabels:
          for i in range(len(self.mixtureLabels [pltIndex])):
            mixtureSplit = self.__splitVariableNames('mixtureLabels', (pltIndex, i))
            if mixtureSplit[2].strip() not in self.sourceData[pltIndex].getVars(mixtureSplit[1].lower()):
              self.raiseAnError(IOError, 'Not found variable "'+ mixtureSplit[2] + '" in "'+mixtureSplit[1]+ '" of DataObject "'+ self.sourceData[pltIndex].name+'"!')
            self.mixtureValues[pltIndex][1].append(np.asarray(dataSet[mixtureSplit[2].strip()].values.astype(float, copy=False)))
        if self.colorMapCoordinates[pltIndex] != None:
          for i in range(len(self.colorMapCoordinates[pltIndex])):
            zSplit = self.__splitVariableNames('colorMap', (pltIndex, i))
            if zSplit[2].strip() not in self.sourceData[pltIndex].getVars(zSplit[1].lower()):
              self.raiseAnError(IOError, 'Not found variable "'+ zSplit[2] + '" in "'+zSplit[1]+ '" of DataObject "'+ self.sourceData[pltIndex].name+'"!')
            self.colorMapValues[pltIndex][1].append(np.asarray(dataSet[zSplit[2].strip()].values.astype(float, copy=False)))
        # check if the array sizes are consistent
        sizeToMatch = self.xValues[pltIndex][1][-1].size
        if self.yCoordinates and self.yValues[pltIndex][1][-1].size != sizeToMatch:
          self.raiseAnError(Exception,"the <y> variable has a size ("+str(self.yValues[pltIndex][1][-1].size)+") that is not consistent with respect the one ("+str(sizeToMatch)+") inputted in <x>")
        if self.zCoordinates and self.dim > 2 and self.zValues[pltIndex][1][-1].size != sizeToMatch:
          self.raiseAnError(Exception,"the <z> variable has a size ("+str(self.zValues[pltIndex][1][-1].size)+") that is not consistent with respect the one ("+str(sizeToMatch)+") inputted in <x>")
        if self.colorMapCoordinates[pltIndex] != None and self.colorMapValues[pltIndex][1][-1].size != sizeToMatch:
          self.raiseAnError(Exception,"the <colorMap> variable has a size ("+str(self.colorMapValues[pltIndex][1][-1].size)+") that is not consistent with respect the one ("+str(sizeToMatch)+") inputted in <x>")
      else:
        # HistorySet
        pivotParam = self.sourceData[pltIndex].indexes[0]
        for cnt in range(len(self.sourceData[pltIndex])):
          maxSize = 0
          for i in range(len(self.xCoordinates [pltIndex])):
            xSplit = self.__splitVariableNames('x', (pltIndex, i))
            outputIndexes = self.sourceData[pltIndex].indexes if xSplit[1].lower() == 'output' else []
            if xSplit[2].strip() not in self.sourceData[pltIndex].getVars(xSplit[1].lower())+outputIndexes:
              self.raiseAnError(IOError, 'Not found variable "'+ xSplit[2] + '" in "'+xSplit[1]+ '" of DataObject "'+ self.sourceData[pltIndex].name+'"!')
            # for variable from input space, it will return array(float), not 1d array
            self.xValues[pltIndex][cnt].append(np.atleast_1d(dataSet.isel({'RAVEN_sample_ID':cnt},False).dropna(pivotParam)[xSplit[2]].values.astype(float, copy=False)))
            maxSize = self.xValues[pltIndex][cnt][-1].size if self.xValues[pltIndex][cnt][-1].size > maxSize else maxSize
          if self.yCoordinates :
            for i in range(len(self.yCoordinates [pltIndex])):
              ySplit = self.__splitVariableNames('y', (pltIndex, i))
              outputIndexes = self.sourceData[pltIndex].indexes if ySplit[1].lower() == 'output' else []
              if ySplit[2].strip() not in self.sourceData[pltIndex].getVars(ySplit[1].lower())+outputIndexes:
                self.raiseAnError(IOError, 'Not found variable "'+ ySplit[2] + '" in "'+ySplit[1]+ '" of DataObject "'+ self.sourceData[pltIndex].name+'"!')
              self.yValues[pltIndex][cnt].append(np.atleast_1d(dataSet.isel({'RAVEN_sample_ID':cnt},False).dropna(pivotParam)[ySplit[2]].values.astype(float, copy=False)))
              maxSize = self.yValues[pltIndex][cnt][-1].size if self.yValues[pltIndex][cnt][-1].size > maxSize else maxSize
          if self.zCoordinates  and self.dim > 2:
            for i in range(len(self.zCoordinates [pltIndex])):
              zSplit = self.__splitVariableNames('z', (pltIndex, i))
              outputIndexes = self.sourceData[pltIndex].indexes if zSplit[1].lower() == 'output' else []
              if zSplit[2].strip() not in self.sourceData[pltIndex].getVars(zSplit[1].lower())+outputIndexes:
                self.raiseAnError(IOError, 'Not found variable "'+ zSplit[2] + '" in "'+zSplit[1]+ '" of DataObject "'+ self.sourceData[pltIndex].name+'"!')
              self.zValues[pltIndex][cnt].append(np.atleast_1d(dataSet.isel({'RAVEN_sample_ID':cnt},False).dropna(pivotParam)[zSplit[2]].values.astype(float, copy=False)))
              maxSize = self.zValues[pltIndex][cnt][-1].size if self.zValues[pltIndex][cnt][-1].size > maxSize else maxSize
          if self.colorMapCoordinates[pltIndex] != None:
            for i in range(len(self.colorMapCoordinates[pltIndex])):
              colorSplit = self.__splitVariableNames('colorMap', (pltIndex, i))
              outputIndexes = self.sourceData[pltIndex].indexes if colorSplit[1].lower() == 'output' else []
              if colorSplit[2].strip() not in self.sourceData[pltIndex].getVars(colorSplit[1].lower())+outputIndexes:
                self.raiseAnError(IOError, 'Not found variable "'+ colorSplit[2] + '" in "'+colorSplit[1]+ '" of DataObject "'+ self.sourceData[pltIndex].name+'"!')
              self.colorMapValues[pltIndex][cnt].append(dataSet.isel({'RAVEN_sample_ID':cnt},False).dropna(pivotParam)[colorSplit[2]].values.astype(float, copy=False))
              maxSize = self.colorMapValues[pltIndex][cnt][-1].size if self.colorMapValues[pltIndex][cnt][-1].size > maxSize else maxSize
          # expand the scalars in case they need to be plotted against histories
          if self.xValues[pltIndex][cnt][-1].size == 1 and maxSize > 1:
            self.xValues[pltIndex][cnt][-1] = np.full(maxSize, self.xValues[pltIndex][cnt][-1])
          if self.yCoordinates and self.yValues[pltIndex][cnt][-1].size == 1 and maxSize > 1:
            self.yValues[pltIndex][cnt][-1] = np.full(maxSize, self.yValues[pltIndex][cnt][-1])
          if self.zCoordinates and self.dim > 2 and self.zValues[pltIndex][cnt][-1].size == 1 and maxSize > 1:
            self.zValues[pltIndex][cnt][-1] = np.full(maxSize, self.zValues[pltIndex][cnt][-1])
          if self.colorMapCoordinates[pltIndex] is not None and self.colorMapValues[pltIndex][cnt][-1].size == 1 and maxSize > 1:
            self.colorMapValues[pltIndex][cnt][-1] = np.full(maxSize, self.colorMapValues[pltIndex][cnt][-1])
          # check if the array sizes are consistent
          if self.yCoordinates and self.yValues[pltIndex][cnt][-1].size != maxSize:
            self.raiseAnError(Exception,"the <y> variable has a size ("+str(self.yValues[pltIndex][cnt][-1].size)+") that is not consistent with respect the one ("+str(sizeToMatch)+") inputted in <x>")
          if self.zCoordinates and self.dim > 2 and self.zValues[pltIndex][cnt][-1].size != maxSize:
            self.raiseAnError(Exception,"the <z> variable has a size ("+str(self.zValues[pltIndex][cnt][-1].size)+") that is not consistent with respect the one ("+str(sizeToMatch)+") inputted in <x>")
          if self.colorMapCoordinates[pltIndex] != None and len(self.colorMapValues[pltIndex][cnt][-1]) != maxSize:
            self.raiseAnError(Exception,"the <colorMap> variable has a size ("+str(self.colorMapValues[pltIndex][cnt][-1].size)+") that is not consistent with respect the one ("+str(sizeToMatch)+") inputted in <x>")

      # check if something has been got or not
      if len(self.xValues[pltIndex].keys()) == 0:
        return False
      else:
        for key in self.xValues[pltIndex].keys():
          if len(self.xValues[pltIndex][key]) == 0:
            return False
          else:
            for i in range(len(self.xValues[pltIndex][key])):
              if self.xValues[pltIndex][key][i].size == 0:
                return False
      if self.yCoordinates :
        if len(self.yValues[pltIndex].keys()) == 0:
          return False
        else:
          for key in self.yValues[pltIndex].keys():
            if len(self.yValues[pltIndex][key]) == 0:
              return False
            else:
              for i in range(len(self.yValues[pltIndex][key])):
                if self.yValues[pltIndex][key][i].size == 0:
                  return False
      if self.zCoordinates  and self.dim > 2:
        if len(self.zValues[pltIndex].keys()) == 0:
          return False
        else:
          for key in self.zValues[pltIndex].keys():
            if len(self.zValues[pltIndex][key]) == 0:
              return False
            else:
              for i in range(len(self.zValues[pltIndex][key])):
                if self.zValues[pltIndex][key][i].size == 0:
                  return False
      if self.clusterLabels :
        if len(self.clusterValues[pltIndex].keys()) == 0:
          return False
        else:
          for key in self.clusterValues[pltIndex].keys():
            if len(self.clusterValues[pltIndex][key]) == 0:
              return False
            else:
              for i in range(len(self.clusterValues[pltIndex][key])):
                if self.clusterValues[pltIndex][key][i].size == 0:
                  return False
      if self.mixtureLabels :
        if len(self.mixtureValues[pltIndex].keys()) == 0:
          return False
        else:
          for key in self.mixtureValues[pltIndex].keys():
            if len(self.mixtureValues[pltIndex][key]) == 0:
              return False
            else:
              for i in range(len(self.mixtureValues[pltIndex][key])):
                if self.mixtureValues[pltIndex][key][i].size == 0:
                  return False
      if self.colorMapCoordinates[pltIndex] != None:
        if len(self.colorMapValues[pltIndex].keys()) == 0:
          return False
        else:
          for key in self.colorMapValues[pltIndex].keys():
            if len(self.colorMapValues[pltIndex][key]) == 0:
              return False
            else:
              for i in range(len(self.colorMapValues[pltIndex][key])):
                if self.colorMapValues[pltIndex][key][i].size == 0:
                  return False
    return True

  def __executeActions(self):
    """
      Function to execute the actions that must be performed on this plot (for
      example, set the x,y,z axis ranges, etc.)
      @ In, None
      @ Out, None
    """
    if 'labelFormat' not in self.options.keys():
      if self.dim == 2:
        plt.gca().yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.ticklabel_format(**{'style':'sci', 'scilimits':(0, 1), 'useOffset':False, 'axis':'both'})
      if self.dim == 3:
        #plt.figure().gca(projection = '3d').yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        #plt.figure().gca(projection = '3d').xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        #plt.figure().gca(projection = '3d').zaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        self.plt3D.ticklabel_format(**{'style':'sci', 'scilimits':(0,1), 'useOffset':False, 'axis':'both'})
    if 'title' not in self.options.keys():
      if self.dim == 2:
        plt.title(self.name, fontdict = {'verticalalignment':'baseline', 'horizontalalignment':'center'})
      if self.dim == 3:
        self.plt3D.set_title(self.name, fontdict = {'verticalalignment':'baseline', 'horizontalalignment':'center'})
    for key in self.options.keys():
      if key in ['how', 'plotSettings', 'figureProperties', 'colorbar']:
        pass
      elif key == 'range':
        if self.dim == 2:
          if 'ymin' in self.options[key].keys():
            plt.ylim(ymin = ast.literal_eval(self.options[key]['ymin']))
          if 'ymax' in self.options[key].keys():
            plt.ylim(ymax = ast.literal_eval(self.options[key]['ymax']))
          if 'xmin' in self.options[key].keys():
            plt.xlim(xmin = ast.literal_eval(self.options[key]['xmin']))
          if 'xmax' in self.options[key].keys():
            plt.xlim(xmax = ast.literal_eval(self.options[key]['xmax']))
        elif self.dim == 3:
          if 'xmin' in self.options[key].keys():
            self.plt3D.set_xlim3d(xmin = ast.literal_eval(self.options[key]['xmin']))
          if 'xmax' in self.options[key].keys():
            self.plt3D.set_xlim3d(xmax = ast.literal_eval(self.options[key]['xmax']))
          if 'ymin' in self.options[key].keys():
            self.plt3D.set_ylim3d(ymin = ast.literal_eval(self.options[key]['ymin']))
          if 'ymax' in self.options[key].keys():
            self.plt3D.set_ylim3d(ymax = ast.literal_eval(self.options[key]['ymax']))
          if 'zmin' in self.options[key].keys():
            self.plt3D.set_zlim3d(bottom = ast.literal_eval(self.options[key]['zmin']))
          if 'zmax' in self.options[key].keys():
            self.plt3D.set_zlim3d(top = ast.literal_eval(self.options[key]['zmax']))

      elif key == 'labelFormat':
        if 'style' not in self.options[key].keys():
          self.options[key]['style'] = 'sci'
        if 'limits' not in self.options[key].keys():
          self.options[key]['limits'] = '(0,0)'
        if 'useOffset' not in self.options[key].keys():
          self.options[key]['useOffset'] = 'False'
        if 'axis' not in self.options[key].keys():
          self.options[key]['axis'] = 'both'
        if self.dim == 2:
          plt.ticklabel_format(**{'style':self.options[key]['style'], 'scilimits':ast.literal_eval(self.options[key]['limits']), 'useOffset':ast.literal_eval(self.options[key]['useOffset']), 'axis':self.options[key]['axis']})
        elif self.dim == 3:
          self.plt3D.ticklabel_format(**{'style':self.options[key]['style'], 'scilimits':ast.literal_eval(self.options[key]['limits']), 'useOffset':ast.literal_eval(self.options[key]['useOffset']), 'axis':self.options[key]['axis']})
      elif key == 'camera':
        if self.dim == 2:
          self.raiseAWarning('2D plots have not a camera attribute... They are 2D!!!!')
        elif self.dim == 3:
          if 'elevation' in self.options[key].keys() and 'azimuth' in self.options[key].keys():
            self.plt3D.view_init(elev = float(self.options[key]['elevation']), azim = float(self.options[key]['azimuth']))
          elif 'elevation' in self.options[key].keys() and 'azimuth' not in self.options[key].keys():
            self.plt3D.view_init(elev = float(self.options[key]['elevation']), azim = None)
          elif 'elevation' not in self.options[key].keys() and 'azimuth' in self.options[key].keys():
            self.plt3D.view_init(elev = None, azim = float(self.options[key]['azimuth']))
      elif key == 'title':
        if self.dim == 2:
          plt.title(self.options[key]['text'], **self.options[key].get('attributes', {}))
        elif self.dim == 3:
          self.plt3D.set_title(self.options[key]['text'], **self.options[key].get('attributes', {}))
      elif key == 'scale':
        if self.dim == 2:
          if 'xscale' in self.options[key].keys():
            plt.xscale(self.options[key]['xscale'], nonposy = 'clip')
          if 'yscale' in self.options[key].keys():
            plt.yscale(self.options[key]['yscale'], nonposy = 'clip')
        elif self.dim == 3:
          if 'xscale' in self.options[key].keys():
            self.plt3D.set_xscale(self.options[key]['xscale'], nonposy = 'clip')
          if 'yscale' in self.options[key].keys():
            self.plt3D.set_yscale(self.options[key]['yscale'], nonposy = 'clip')
          if 'zscale' in self.options[key].keys():
            self.plt3D.set_zscale(self.options[key]['zscale'], nonposy = 'clip')
      elif key == 'addText':
        if 'position' not in self.options[key].keys():
          if self.dim == 2:
            self.options[key]['position'] = '0.0,0.0'
          else:
            self.options[key]['position'] = '0.0,0.0,0.0'
        if 'withdash' not in self.options[key].keys():
          self.options[key]['withdash'] = 'False'
        if 'fontdict' not in self.options[key].keys():
          self.options[key]['fontdict'] = 'None'
        else:
          try:
            tempVar = ast.literal_eval(self.options[key]['fontdict'])
            self.options[key]['fontdict'] = str(tempVar)
          except AttributeError:
            self.raiseAnError(TypeError, 'In ' + key + ' tag: can not convert the string "' + self.options[key]['fontdict'] + '" to a dictionary! Check syntax for python function ast.literal_eval')
        if self.dim == 2 :
          plt.text(float(self.options[key]['position'].split(',')[0]), float(self.options[key]['position'].split(',')[1]), self.options[key]['text'], fontdict = ast.literal_eval(self.options[key]['fontdict']), **self.options[key].get('attributes', {}))
        elif self.dim == 3:
          self.plt3D.text(float(self.options[key]['position'].split(',')[0]), float(self.options[key]['position'].split(',')[1]), float(self.options[key]['position'].split(',')[2]), self.options[key]['text'], fontdict = ast.literal_eval(self.options[key]['fontdict']), **self.options[key].get('attributes', {}))
      elif key == 'autoscale':
        if 'enable' not in self.options[key].keys():
          self.options[key]['enable'] = 'True'
        elif utils.stringIsTrue(self.options[key]['enable']):
          self.options[key]['enable'] = 'True'
        elif utils.stringIsFalse(self.options[key]['enable']):
          self.options[key]['enable'] = 'False'
        if 'axis' not in self.options[key].keys():
          self.options[key]['axis'] = 'both'
        if 'tight' not in self.options[key].keys():
          self.options[key]['tight'] = 'None'

        if self.dim == 2:
          plt.autoscale(enable = ast.literal_eval(self.options[key]['enable']), axis = self.options[key]['axis'], tight = ast.literal_eval(self.options[key]['tight']))
        elif self.dim == 3:
          self.plt3D.autoscale(enable = ast.literal_eval(self.options[key]['enable']), axis = self.options[key]['axis'], tight = ast.literal_eval(self.options[key]['tight']))
      elif key == 'horizontalLine':
        if self.dim == 3:
          self.raiseAWarning('horizontalLine not available in 3-D plots!!')
        elif self.dim == 2:
          if 'y' not in self.options[key].keys():
            self.options[key]['y'] = '0'
          if 'xmin' not in self.options[key].keys():
            self.options[key]['xmin'] = '0'
          if 'xmax' not in self.options[key].keys():
            self.options[key]['xmax'] = '1'
          if 'hold' not in self.options[key].keys():
            self.options[key]['hold'] = 'None'
          plt.axhline(y = ast.literal_eval(self.options[key]['y']), xmin = ast.literal_eval(self.options[key]['xmin']), xmax = ast.literal_eval(self.options[key]['xmax']), hold = ast.literal_eval(self.options[key]['hold']), **self.options[key].get('attributes', {}))
      elif key == 'verticalLine':
        if self.dim == 3:
          self.raiseAWarning('verticalLine not available in 3-D plots!!')
        elif self.dim == 2:
          if 'x' not in self.options[key].keys():
            self.options[key]['x'] = '0'
          if 'ymin' not in self.options[key].keys():
            self.options[key]['ymin'] = '0'
          if 'ymax' not in self.options[key].keys():
            self.options[key]['ymax'] = '1'
          if 'hold' not in self.options[key].keys():
            self.options[key]['hold'] = 'None'
          plt.axvline(x = ast.literal_eval(self.options[key]['x']), ymin = ast.literal_eval(self.options[key]['ymin']), ymax = ast.literal_eval(self.options[key]['ymax']), hold = ast.literal_eval(self.options[key]['hold']), **self.options[key].get('attributes', {}))
      elif key == 'horizontalRectangle':
        if self.dim == 3:
          self.raiseAWarning('horizontalRectangle not available in 3-D plots!!')
        elif self.dim == 2:
          if 'ymin' not in self.options[key].keys():
            self.raiseAnError(IOError, 'ymin parameter is needed for function horizontalRectangle!!')
          if 'ymax' not in self.options[key].keys():
            self.raiseAnError(IOError, 'ymax parameter is needed for function horizontalRectangle!!')
          if 'xmin' not in self.options[key].keys():
            self.options[key]['xmin'] = '0'
          if 'xmax' not in self.options[key].keys():
            self.options[key]['xmax'] = '1'
          plt.axhspan(ast.literal_eval(self.options[key]['ymin']), ast.literal_eval(self.options[key]['ymax']), xmin = ast.literal_eval(self.options[key]['xmin']), xmax = ast.literal_eval(self.options[key]['xmax']), **self.options[key].get('attributes', {}))
      elif key == 'verticalRectangle':
        if self.dim == 3:
          self.raiseAWarning('vertical_rectangle not available in 3-D plots!!')
        elif self.dim == 2:
          if 'xmin' not in self.options[key].keys():
            self.raiseAnError(IOError, 'xmin parameter is needed for function verticalRectangle!!')
          if 'xmax' not in self.options[key].keys():
            self.raiseAnError(IOError, 'xmax parameter is needed for function verticalRectangle!!')
          if 'ymin' not in self.options[key].keys():
            self.options[key]['ymin'] = '0'
          if 'ymax' not in self.options[key].keys():
            self.options[key]['ymax'] = '1'
          plt.axvspan(ast.literal_eval(self.options[key]['xmin']), ast.literal_eval(self.options[key]['xmax']), ymin = ast.literal_eval(self.options[key]['ymin']), ymax = ast.literal_eval(self.options[key]['ymax']), **self.options[key].get('attributes', {}))
      elif key == 'axesBox':
        if   self.dim == 3:
          self.raiseAWarning('axesBox not available in 3-D plots!!')
        elif self.dim == 2:
          plt.box(self.options[key][key])
      elif key == 'axis':
        plt.axis(self.options[key][key])
      elif key == 'grid':
        if 'b' not in self.options[key].keys():
          self.options[key]['b'] = 'off'
        if utils.stringIsTrue(self.options[key]['b']):
          self.options[key]['b'] = 'on'
        elif utils.stringIsFalse(self.options[key]['b']):
          self.options[key]['b'] = 'off'
        if 'which' not in self.options[key].keys():
          self.options[key]['which'] = 'major'
        if 'axis' not in self.options[key].keys():
          self.options[key]['axis'] = 'both'
        if self.dim == 2:
          plt.grid(b = self.options[key]['b'], which = self.options[key]['which'], axis = self.options[key]['axis'], **self.options[key].get('attributes', {}))
        elif self.dim == 3:
          self.plt3D.grid(b = self.options[key]['b'], **self.options[key].get('attributes', {}))
      else:
        self.raiseAWarning('Try to perform not-predifined action ' + key + '. If it does not work check manual and/or relavite matplotlib method specification.')
        kwargs = {}
        for kk in self.options[key]:
          if kk != 'attributes' and kk != key:
            try:
              kwargs[kk] = ast.literal_eval(self.options[key][kk])
            except ValueError:
              kwargs[kk] = self.options[key][kk]
        try:
          if self.dim == 2:
            customFunctionCall = getattr(plt, key)
          else:
            customFunctionCall = getattr(self.plt3D, key)
          self.actPlot = customFunctionCall(**kwargs)
        except AttributeError as ae:
          self.raiseAnError(RuntimeError, '<' + str(ae) + '> -> in execution custom action "' + key + '" in Plot ' + self.name + '.\n ' + self.printTag + ' command has been called in the following way: ' + 'plt.' + key + '(**' + str(kwargs) + ')')

  ####################
  #  PUBLIC METHODS  #
  ####################
  def getInitParams(self):
    """
      This function is called from the base class to print some of the
      information inside the class. Whatever is permanent in the class and not
      inherited from the parent class should be mentioned here. The information
      is passed back in the dictionary. No information about values that change
      during the simulation are allowed.
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = super().getInitParams()
    paramDict[f'OutStream Available {self.dim}D   :'] = self.availableOutStreamTypes[self.dim]
    paramDict['Plot is '] = str(self.dim) + 'D'
    for index in range(len(self.sourceName)):
      paramDict['Source Name ' + str(index) + ' :'] = self.sourceName[index]
    return paramDict

  def endInstructions(self, instructionString):
    """
      Method to execute instructions at end of a step (this is applied when an
      interactive mode is activated)
      @ In, instructionString, string, the instruction to execute
      @ Out, None
    """
    if instructionString == 'interactive' and 'screen' in self.destinations and display:
      self.fig = plt.figure(self.name)
      ## This seems a bit hacky, but we need the ginput in order to block
      ## execution of raven until this is over, however closing the window can
      ## cause this thing to fail.
      try:
        self.fig.ginput(n = -1, timeout = 0, show_clicks = False)
      except Exception as e:
        ## I know this is bad, but it is a single line of code outside our
        ## control, if it fails for any reason it should not be a huge deal, we
        ## just want RAVEN to continue on its merry way when a figure closes.
        self.raiseAWarning('There was an error with figure.ginput:\n', e)
        self.raiseAWarning('... continuing anyway ...')
        pass
      ## We may want to catch a more generic exception since this may be depedent
      ## on the backend used, hence the code replacement above
      # except _tkinter.TclError:
      #   pass

  def initialize(self, inDict):
    """
      Function called to initialize the OutStream, linking it to the proper data
      @ In, inDict, dict, dictionary that contains all the instantiaced classes
        needed for the actual step. The data is looked for in this dictionary
      @ Out, None
    """
    self.xCoordinates = []
    self.sourceName = []

    self.destinations = self.options['how']['how'].lower().split(',')

    if 'figureProperties' in self.options.keys():
      key = 'figureProperties'
      if 'figsize' not in self.options[key].keys():
        self.options[key]['figsize'  ] = None
      else:
        self.options[key]['figsize'  ] = tuple([float(elm) for elm in  ast.literal_eval(self.options[key]['figsize'])])
      if 'dpi' not in self.options[key].keys():
        self.options[key]['dpi'      ] = 'None'
      if 'facecolor' not in self.options[key].keys():
        self.options[key]['facecolor'] = 'None'
      if 'edgecolor' not in self.options[key].keys():
        self.options[key]['edgecolor'] = 'None'
      if 'frameon' not in self.options[key].keys():
        self.options[key]['frameon'  ] = 'True'
      elif utils.stringIsTrue(self.options[key]['frameon']):
        self.options[key]['frameon'] = 'True'
      elif utils.stringIsFalse(self.options[key]['frameon']):
        self.options[key]['frameon'] = 'False'
      self.fig = plt.figure(self.name, figsize = self.options[key]['figsize'], dpi = ast.literal_eval(self.options[key]['dpi']), facecolor = self.options[key]['facecolor'], edgecolor = self.options[key]['edgecolor'], frameon = ast.literal_eval(self.options[key]['frameon']), **self.options[key].get('attributes', {}))
    else:
      self.fig = plt.figure(self.name)

    if 'screen' in self.destinations and display:
      self.fig.show()

    if self.dim == 3:
      self.plt3D = self.fig.add_subplot(111, projection = '3d')

    for pltIndex in range(len(self.options['plotSettings']['plot'])):
      self.colorMapCoordinates[pltIndex] = None
      if 'y' in self.options['plotSettings']['plot'][pltIndex].keys():
        self.yCoordinates = []
      if 'z' in self.options['plotSettings']['plot'][pltIndex].keys():
        self.zCoordinates = []
      if 'clusterLabels' in self.options['plotSettings']['plot'][pltIndex].keys():
        self.clusterLabels = []
      if 'mixtureLabels' in self.options['plotSettings']['plot'][pltIndex].keys():
        self.mixtureLabels = []
      if 'attributes' in self.options['plotSettings']['plot'][pltIndex].keys():
        if 'mixtureMeans' in self.options['plotSettings']['plot'][pltIndex]['attributes'].keys():
          self.mixtureMeans = []
        if 'mixtureCovars' in self.options['plotSettings']['plot'][pltIndex]['attributes'].keys():
          self.mixtureCovars = []
      # if 'colorMap' in self.options['plotSettings']['plot'][pltIndex].keys(): self.colorMapCoordinates = {}
    for pltIndex in range(len(self.options['plotSettings']['plot'])):
      self.xCoordinates.append(self.options['plotSettings']['plot'][pltIndex]['x'].split(','))
      self.sourceName.append(self.xCoordinates [pltIndex][0].split('|')[0].strip())
      if 'y' in self.options['plotSettings']['plot'][pltIndex].keys():
        self.yCoordinates .append(self.options['plotSettings']['plot'][pltIndex]['y'].split(','))
        if self.yCoordinates [pltIndex][0].split('|')[0] != self.sourceName[pltIndex]:
          self.raiseAnError(IOError, 'Every plot can be linked to one Data only. x_cord source is ' + self.sourceName[pltIndex] + '. Got y_cord source is' + self.yCoordinates [pltIndex][0].split('|')[0])
      if 'z' in self.options['plotSettings']['plot'][pltIndex].keys():
        self.zCoordinates .append(self.options['plotSettings']['plot'][pltIndex]['z'].split(','))
        if self.zCoordinates [pltIndex][0].split('|')[0] != self.sourceName[pltIndex]:
          self.raiseAnError(IOError, 'Every plot can be linked to one Data only. x_cord source is ' + self.sourceName[pltIndex] + '. Got z_cord source is' + self.zCoordinates [pltIndex][0].split('|')[0])
      if 'clusterLabels' in self.options['plotSettings']['plot'][pltIndex].keys():
        self.clusterLabels .append(self.options['plotSettings']['plot'][pltIndex]['clusterLabels'].split(','))
        if self.clusterLabels [pltIndex][0].split('|')[0] != self.sourceName[pltIndex]:
          self.raiseAnError(IOError, 'Every plot can be linked to one Data only. x_cord source is ' + self.sourceName[pltIndex] + '. Got clusterLabels source is' + self.clusterLabels [pltIndex][0].split('|')[0])
      if 'mixtureLabels' in self.options['plotSettings']['plot'][pltIndex].keys():
        self.mixtureLabels .append(self.options['plotSettings']['plot'][pltIndex]['mixtureLabels'].split(','))
        if self.mixtureLabels [pltIndex][0].split('|')[0] != self.sourceName[pltIndex]:
          self.raiseAnError(IOError, 'Every plot can be linked to one Data only. x_cord source is ' + self.sourceName[pltIndex] + '. Got mixtureLabels source is' + self.mixtureLabels [pltIndex][0].split('|')[0])
      if 'colorMap' in self.options['plotSettings']['plot'][pltIndex].keys():
        self.colorMapCoordinates[pltIndex] = self.options['plotSettings']['plot'][pltIndex]['colorMap'].split(',')
        # self.colorMapCoordinates.append(self.options['plotSettings']['plot'][pltIndex]['colorMap'].split(','))
        if self.colorMapCoordinates[pltIndex][0].split('|')[0] != self.sourceName[pltIndex]:
          self.raiseAnError(IOError, 'Every plot can be linked to one Data only. x_cord source is ' + self.sourceName[pltIndex] + '. Got colorMap_coordinates source is' + self.colorMapCoordinates[pltIndex][0].split('|')[0])
      for pltIndex in range(len(self.options['plotSettings']['plot'])):
        if 'interpPointsY' not in self.options['plotSettings']['plot'][pltIndex].keys():
          self.options['plotSettings']['plot'][pltIndex]['interpPointsY'] = '20'
        if 'interpPointsX' not in self.options['plotSettings']['plot'][pltIndex].keys():
          self.options['plotSettings']['plot'][pltIndex]['interpPointsX'] = '20'
        if 'interpolationType' not in self.options['plotSettings']['plot'][pltIndex].keys():
          self.options['plotSettings']['plot'][pltIndex]['interpolationType'] = 'linear'
        elif self.options['plotSettings']['plot'][pltIndex]['interpolationType'] not in self.availableInterpolators:
          self.raiseAnError(IOError, 'surface interpolation unknown. Available are :' + str(self.availableInterpolators))
        if 'epsilon' not in self.options['plotSettings']['plot'][pltIndex].keys():
          self.options['plotSettings']['plot'][pltIndex]['epsilon'] = '2'
        if 'smooth' not in self.options['plotSettings']['plot'][pltIndex].keys():
          self.options['plotSettings']['plot'][pltIndex]['smooth'] = '0.0'
        if 'cmap' not in self.options['plotSettings']['plot'][pltIndex].keys():
          self.options['plotSettings']['plot'][pltIndex]['cmap'] = 'None'
        #    else:             self.options['plotSettings']['plot'][pltIndex]['cmap'] = 'jet'
        elif self.options['plotSettings']['plot'][pltIndex]['cmap'] is not 'None' and self.options['plotSettings']['plot'][pltIndex]['cmap'] not in matplotlib.cm.datad.keys():
          raise('ERROR. The colorMap you specified does not exist... Available are ' + str(matplotlib.cm.datad.keys()))
        if 'interpolationTypeBackUp' not in self.options['plotSettings']['plot'][pltIndex].keys():
          self.options['plotSettings']['plot'][pltIndex]['interpolationTypeBackUp'] = 'nearest'
        elif self.options['plotSettings']['plot'][pltIndex]['interpolationTypeBackUp'] not in self.availableInterpolators:
          self.raiseAnError(IOError, 'surface interpolation (BackUp) unknown. Available are :' + str(self.availableInterpolators))
      if 'attributes' in self.options['plotSettings']['plot'][pltIndex].keys():
        if 'mixtureMeans' in self.options['plotSettings']['plot'][pltIndex]['attributes'].keys():
          self.mixtureMeans.append(self.options['plotSettings']['plot'][pltIndex]['attributes']['mixtureMeans'].split(','))
        if 'mixtureCovars' in self.options['plotSettings']['plot'][pltIndex]['attributes'].keys():
          self.mixtureCovars.append(self.options['plotSettings']['plot'][pltIndex]['attributes']['mixtureCovars'].split(','))
    self.numberAggregatedOS = len(self.options['plotSettings']['plot'])
    # collect sources
    self.legacyCollectSources(inDict)
    # initialize here the base class
    super().initialize(inDict)
    # execute actions (we execute the actions here also because we can perform a check at runtime!!
    self.__executeActions()

  def handleInput(self, xmlNode):
    """
      This Function is called from the base class, It reads the parameters that
      belong to a plot block
      Overriding default methods, until this interface uses input params. FIXME
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    # because we're reading XML not inputParams, we don't call super, and have to set our own:
    # - name
    # - subdirectory
    # - overwrite
    self.name = xmlNode.attrib['name']
    subDir = xmlNode.attrib.get('dir', None)
    if subDir:
      subDir = os.path.expanduser(subDir)
    self.subDirectory = subDir
    if 'overwrite' in xmlNode.attrib:
      self.overwrite = utils.stringIsTrue(xmlNode.attrib['overwrite'])
    foundPlot = False
    if 'dim' in xmlNode.attrib:
      self.raiseAnError(IOError,"the 'dim' attribute has been deprecated. This warning became an error in January 2017")
    for subnode in xmlNode:
      # if actions, read actions block
      if subnode.tag == 'filename':
        self.filename = subnode.text
      if subnode.tag in ['actions']:
        self.__readPlotActions(subnode)
      if subnode.tag in ['plotSettings']:
        self.options[subnode.tag] = {}
        self.options[subnode.tag]['plot'] = []
        for subsub in subnode:
          if subsub.tag == 'gridSpace':
            # if self.dim == 3: self.raiseAnError(IOError, 'SubPlot option can not be used with 3-dimensional plots!')
            self.options[subnode.tag][subsub.tag] = subsub.text.strip()
          elif subsub.tag == 'plot':
            tempDict = {}
            foundPlot = True
            for subsubsub in subsub:
              if subsubsub.tag == 'gridLocation':
                tempDict[subsubsub.tag] = {}
                for subsubsubsub in subsubsub:
                  tempDict[subsubsub.tag][subsubsubsub.tag] = subsubsubsub.text.strip()
              elif subsubsub.tag == 'range':
                tempDict[subsubsub.tag] = {}
                for subsubsubsub in subsubsub:
                  tempDict[subsubsub.tag][subsubsubsub.tag] = subsubsubsub.text.strip()
              elif subsubsub.tag != 'kwargs':
                tempDict[subsubsub.tag] = subsubsub.text.strip()
              else:
                tempDict['attributes'] = {}
                for sss in subsubsub:
                  tempDict['attributes'][sss.tag] = sss.text.strip()
            self.options[subnode.tag][subsub.tag].append(tempDict)
          elif subsub.tag == 'legend':
            self.options[subnode.tag][subsub.tag] = {}
            for legendChild in subsub:
              self.options[subnode.tag][subsub.tag][legendChild.tag] = utils.tryParse(legendChild.text.strip())
          else:
            self.options[subnode.tag][subsub.tag] = subsub.text.strip()
      if subnode.tag in 'title':
        self.options[subnode.tag] = {}
        for subsub in subnode:
          self.options[subnode.tag][subsub.tag] = subsub.text.strip()
        if 'text'     not in self.options[subnode.tag].keys():
          self.options[subnode.tag]['text'    ] = xmlNode.attrib['name']
        if 'location' not in self.options[subnode.tag].keys():
          self.options[subnode.tag]['location'] = 'center'
      ## is this 'figureProperties' valid?
      if subnode.tag == 'figureProperties':
        self.options[subnode.tag] = {}
        for subsub in subnode:
          self.options[subnode.tag][subsub.tag] = subsub.text.strip()
    self.type = 'OutStreamPlot'

    if not 'plotSettings' in self.options.keys():
      self.raiseAnError(IOError, 'For plot named ' + self.name + ' the plotSettings block is required.')

    if not foundPlot:
      self.raiseAnError(IOError, 'For plot named' + self.name + ', No plot section has been found in the plotSettings block!')

    self.outStreamTypes = []
    xyz, xy             = sorted(['x','y','z']), sorted(['x','y'])
    for pltIndex in range(len(self.options['plotSettings']['plot'])):
      if not 'type' in self.options['plotSettings']['plot'][pltIndex].keys():
        self.raiseAnError(IOError, 'For plot named' + self.name + ', No plot type keyword has been found in the plotSettings/plot block!')
      else:
        # check the dimension and check the consistency
        if set(xyz) < set(self.options['plotSettings']['plot'][pltIndex].keys()):
          dim = 3
        elif set(xy) < set(self.options['plotSettings']['plot'][pltIndex].keys()):
          dim = 2 if self.options['plotSettings']['plot'][pltIndex]['type'] != 'histogram' else 3
        elif set(['x']) < set(self.options['plotSettings']['plot'][pltIndex].keys()) and self.options['plotSettings']['plot'][pltIndex]['type'] == 'histogram':
          dim = 2
        else:
          self.raiseAnError(IOError, 'Wrong dimensionality or axis specification for plot '+self.name+'.')
        if self.dim is not None and self.dim != dim:
          self.raiseAnError(IOError, 'The OutStream Plot '+self.name+' combines 2D and 3D plots. This is not supported!')
        self.dim = dim
        if self.availableOutStreamTypes[self.dim].count(self.options['plotSettings']['plot'][pltIndex]['type']) == 0:
          self.raiseAMessage('For plot named' + self.name + ', type ' + self.options['plotSettings']['plot'][pltIndex]['type'] + ' is not among pre-defined plots! \n The OutstreamSystem will try to construct a call on the fly!', 'ExceptedError')
        self.outStreamTypes.append(self.options['plotSettings']['plot'][pltIndex]['type'])
    self.raiseADebug('matplotlib version is ' + str(matplotlib.__version__))

    if self.dim not in [2, 3]:
      self.raiseAnError(TypeError, 'This Plot interface is able to handle 2D-3D plot only')

    if 'gridSpace' in self.options['plotSettings'].keys():
      grid = list(map(int, self.options['plotSettings']['gridSpace'].split(' ')))
      self.gridSpace = matplotlib.gridspec.GridSpec(grid[0], grid[1])

  def run(self):
    """
      Function to show and/or save a plot (outputs Plot on the screen or on file/s)
      @ In,  None
      @ Out, None
    """
    # fill the x_values,y_values,z_values dictionaries
    if not self.__fillCoordinatesFromSource():
      self.raiseAWarning('Nothing to Plot Yet. Returning.')
      return
    # reactivate the figure
    self.fig = plt.figure(self.name)
    self.counter += 1
    if self.counter > 1:
      self.fig.clear()
      self.actcm = None
    # start plotting.... we are here fort that...aren't we?
    # loop over the plots that need to be included in this figure
    from copy import deepcopy
    clusterDict = deepcopy(self.outStreamTypes)

    for pltIndex in range(len(self.outStreamTypes)):
      plotSettings = self.options['plotSettings']['plot'][pltIndex]

      if 'gridLocation' in plotSettings.keys():
        x = None
        y = None
        if 'x' in  plotSettings['gridLocation'].keys():
          x = list(map(int, plotSettings['gridLocation']['x'].strip().split(' ')))
        else:
          x = None
        if 'y' in  plotSettings['gridLocation'].keys():
          y = list(map(int, plotSettings['gridLocation']['y'].strip().split(' ')))
        else:
          y = None
        if   (len(x) == 1 and len(y) == 1):
          if self.dim == 2:
            plt.subplot(self.gridSpace[x[0], y[0]])
          else:
            self.plt3D = plt.subplot(self.gridSpace[x[0], y[0]], projection = '3d')
        elif (len(x) == 1 and len(y) != 1):
          if self.dim == 2:
            plt.subplot(self.gridSpace[x[0], y[0]:y[-1]])
          else:
            self.plt3D = plt.subplot(self.gridSpace[x[0], y[0]:y[-1]], projection = '3d')
        elif (len(x) != 1 and len(y) == 1):
          if self.dim == 2:
            plt.subplot(self.gridSpace[x[0]:x[-1], y[0]])
          else:
            self.plt3D = plt.subplot(self.gridSpace[x[0]:x[-1], y[0]], projection = '3d')
        else:
          if self.dim == 2:
            plt.subplot(self.gridSpace[x[0]:x[-1], y[0]:y[-1]])
          else:
            self.plt3D = plt.subplot(self.gridSpace[x[0]:x[-1], y[0]:y[-1]], projection = '3d')

      # calling "plt.hold" has been deprecated.
      # If the figure isn't cleared (or a new figure opened), it will keep adding plots.
      # This set of comments can be removed when "hold" has been demonstrated unneccessary.
      # if len(self.outStreamTypes) > 1:
      #  plt.hold(True)
      if 'gridSpace' in self.options['plotSettings'].keys():
        plt.locator_params(axis = 'y', nbins = 4)
        plt.locator_params(axis = 'x', nbins = 2)
        #plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 1))
        if 'range' in plotSettings.keys():
          axes_range = plotSettings['range']
          if self.dim == 2:
            if 'ymin' in axes_range.keys():
              plt.ylim(ymin = ast.literal_eval(axes_range['ymin']))
            if 'ymax' in axes_range.keys():
              plt.ylim(ymax = ast.literal_eval(axes_range['ymax']))
            if 'xmin' in axes_range.keys():
              plt.xlim(xmin = ast.literal_eval(axes_range['xmin']))
            if 'xmax' in axes_range.keys():
              plt.xlim(xmax = ast.literal_eval(axes_range['xmax']))
          elif self.dim == 3:
            if 'xmin' in axes_range.keys():
              self.plt3D.set_xlim3d(xmin = ast.literal_eval(axes_range['xmin']))
            if 'xmax' in axes_range.keys():
              self.plt3D.set_xlim3d(xmax = ast.literal_eval(axes_range['xmax']))
            if 'ymin' in axes_range.keys():
              self.plt3D.set_ylim3d(ymin = ast.literal_eval(axes_range['ymin']))
            if 'ymax' in axes_range.keys():
              self.plt3D.set_ylim3d(ymax = ast.literal_eval(axes_range['ymax']))
            if 'zmin' in axes_range.options['plotSettings']['plot'][pltIndex].keys():
              if 'zmax' not in axes_range.options['plotSettings'].keys():
                self.raiseAWarning('zmin inputted but not zmax. zmin ignored! ')
              else:
                self.plt3D.set_zlim(ast.literal_eval(axes_range['zmin']), ast.literal_eval(self.options['plotSettings']['zmax']))
            if 'zmax' in axes_range.keys():
              if 'zmin' not in axes_range.keys():
                self.raiseAWarning('zmax inputted but not zmin. zmax ignored! ')
              else:
                self.plt3D.set_zlim(ast.literal_eval(axes_range['zmin']), ast.literal_eval(axes_range['zmax']))
        if 'xlabel' not in plotSettings.keys():
          if   self.dim == 2:
            plt.xlabel('x')
          elif self.dim == 3:
            self.plt3D.set_xlabel('x')
        else:
          if   self.dim == 2:
            plt.xlabel(plotSettings['xlabel'])
          elif self.dim == 3:
            self.plt3D.set_xlabel(plotSettings['xlabel'])
        if 'ylabel' not in plotSettings.keys():
          if   self.dim == 2:
            plt.ylabel('y')
          elif self.dim == 3:
            self.plt3D.set_ylabel('y')
        else:
          if   self.dim == 2:
            plt.ylabel(plotSettings['ylabel'])
          elif self.dim == 3:
            self.plt3D.set_ylabel(plotSettings['ylabel'])
        if 'zlabel' in plotSettings.keys():
          if   self.dim == 2:
            self.raiseAWarning('zlabel keyword does not make sense in 2-D Plots!')
          elif self.dim == 3 and self.zCoordinates:
            self.plt3D.set_zlabel(plotSettings['zlabel'])
        elif self.dim == 3 and self.zCoordinates:
          self.plt3D.set_zlabel('z')
      else:
        if 'xlabel' not in self.options['plotSettings'].keys():
          if   self.dim == 2:
            plt.xlabel('x')
          elif self.dim == 3:
            self.plt3D.set_xlabel('x')
        else:
          if   self.dim == 2:
            plt.xlabel(self.options['plotSettings']['xlabel'])
          elif self.dim == 3:
            self.plt3D.set_xlabel(self.options['plotSettings']['xlabel'])
        if 'ylabel' not in self.options['plotSettings'].keys():
          if   self.dim == 2:
            plt.ylabel('y')
          elif self.dim == 3:
            self.plt3D.set_ylabel('y')
        else:
          if   self.dim == 2:
            plt.ylabel(self.options['plotSettings']['ylabel'])
          elif self.dim == 3:
            self.plt3D.set_ylabel(self.options['plotSettings']['ylabel'])
        if 'zlabel' in self.options['plotSettings'].keys():
          if   self.dim == 2:
            self.raiseAWarning('zlabel keyword does not make sense in 2-D Plots!')
          elif self.dim == 3 and self.zCoordinates:
            self.plt3D.set_zlabel(self.options['plotSettings']['zlabel'])
        elif self.dim == 3 and self.zCoordinates:
          self.plt3D.set_zlabel('z')

      if 'legend' in self.options['plotSettings']:
        if 'label' not in plotSettings.get('attributes', {}):
          if 'attributes' not in plotSettings:
            plotSettings['attributes'] = {}
          plotSettings['attributes']['label'] = self.outStreamTypes[pltIndex] + ' ' + str(pltIndex)

      # Let's start plotting
      #################
      #  SCATTER PLOT #
      #################
      self.raiseADebug('creating plot ' + self.name)
      if self.outStreamTypes[pltIndex] == 'scatter':
        if 's' not in plotSettings.keys():
          plotSettings['s'] = '20'
        if 'c' not in plotSettings.keys():
          plotSettings['c'] = 'b'
        if 'marker' not in plotSettings.keys():
          plotSettings['marker'] = 'o'
        if 'alpha' not in plotSettings.keys():
          plotSettings['alpha'] = 'None'
        if 'linewidths' not in plotSettings.keys():
          plotSettings['linewidths'] = 'None'
        if self.colorMapCoordinates[pltIndex] != None:
          #Find the max and min colormap values for the whole set.
          firstKey = utils.first(self.xValues[pltIndex].keys())
          vmin = np.amin(self.colorMapValues[pltIndex][firstKey])
          vmax = np.amax(self.colorMapValues[pltIndex][firstKey])
          for key in self.xValues[pltIndex].keys():
            vmin = min(vmin,np.amin(self.colorMapValues[pltIndex][key]))
            vmax = max(vmax,np.amax(self.colorMapValues[pltIndex][key]))
          plotSettings['norm'] = matplotlib.colors.Normalize(vmin,vmax)
        for key in self.xValues[pltIndex].keys():
          for xIndex in range(len(self.xValues[pltIndex][key])):
            for yIndex in range(len(self.yValues[pltIndex][key])):
              scatterPlotOptions = {'s':ast.literal_eval(plotSettings['s']),
                                    'marker':(plotSettings['marker']),
                                    'alpha':ast.literal_eval(plotSettings['alpha']),
                                    'linewidths':ast.literal_eval(plotSettings['linewidths'])}
              if self.colorMapCoordinates[pltIndex] != None:
                scatterPlotOptions['norm'] = plotSettings['norm']
              scatterPlotOptions.update(plotSettings.get('attributes', {}))
              if self.dim == 2:
                if self.colorMapCoordinates[pltIndex] != None:
                  scatterPlotOptions['c'] = self.colorMapValues[pltIndex][key][xIndex]
                  scatterPlotOptions['cmap'] = matplotlib.cm.get_cmap("winter")
                  if self.actcm:
                    first = False
                  else:
                    first = True
                  if plotSettings['cmap'] == 'None':
                    #if plotSettings['cmap'] == 'None': plotSettings['cmap'] = 'winter'
                    self.actPlot = plt.scatter(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex], **scatterPlotOptions)
                    if 'colorbar' not in self.options.keys() or self.options['colorbar']['colorbar'] != 'off':
                      if first:
                        m = matplotlib.cm.ScalarMappable(cmap = self.actPlot.cmap, norm = self.actPlot.norm)
                        m.set_array(self.colorMapValues[pltIndex][key])
                        self.actcm = self.fig.colorbar(m)
                        self.actcm.set_label(self.colorMapCoordinates[pltIndex][0].split('|')[-1].replace(')', ''))
                      else:
                        try:
                          self.actcm.draw_all()
                        except:
                          m = matplotlib.cm.ScalarMappable(cmap = self.actPlot.cmap, norm = self.actPlot.norm)
                          m.set_array(self.colorMapValues[pltIndex][key])
                          self.actcm = self.fig.colorbar(m)
                          self.actcm.set_label(self.colorMapCoordinates[pltIndex][0].split('|')[-1].replace(')', ''))
                  else:
                    scatterPlotOptions['cmap'] = plotSettings['cmap']
                    self.actPlot = plt.scatter(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex], **scatterPlotOptions)
                    if 'colorbar' not in self.options.keys() or self.options['colorbar']['colorbar'] != 'off':
                      if first:
                        m = matplotlib.cm.ScalarMappable(cmap = self.actPlot.cmap, norm = self.actPlot.norm)
                        m.set_array(self.colorMapValues[pltIndex][key])
                        self.actcm = self.fig.colorbar(m)
                        self.actcm.set_label(self.colorMapCoordinates[pltIndex][0].split('|')[-1].replace(')', ''))
                      else:
                        m = matplotlib.cm.ScalarMappable(cmap = self.actPlot.cmap, norm = self.actPlot.norm)
                        m.set_clim(vmin = min(self.colorMapValues[pltIndex][key][-1]), vmax = max(self.colorMapValues[pltIndex][key][-1]))
                        self.actcm.draw_all()
                else:
                  if 'color' not in scatterPlotOptions:
                    scatterPlotOptions['c'] = plotSettings['c']
                  self.actPlot = plt.scatter(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex], **scatterPlotOptions)
              elif self.dim == 3:
                for zIndex in range(len(self.zValues[pltIndex][key])):
                  if self.colorMapCoordinates[pltIndex] != None:
                    scatterPlotOptions['c'] = self.colorMapValues[pltIndex][key][zIndex]
                    if self.actcm:
                      first = False
                    else:
                      first = True
                    if plotSettings['cmap'] == 'None':
                      self.actPlot = self.plt3D.scatter(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex], self.zValues[pltIndex][key][zIndex], **scatterPlotOptions)
                      if 'colorbar' not in self.options.keys() or self.options['colorbar']['colorbar'] != 'off':
                        if first:
                          m = matplotlib.cm.ScalarMappable(cmap = self.actPlot.cmap, norm = self.actPlot.norm)
                          m.set_array(self.colorMapValues[pltIndex][key])
                          self.actcm = self.fig.colorbar(m)
                          self.actcm.set_label(self.colorMapCoordinates[pltIndex][0].split('|')[-1].replace(')', ''))
                        else:
                          self.actcm.draw_all()
                    else:
                      scatterPlotOptions['cmap'] = plotSettings['cmap']
                      self.actPlot = self.plt3D.scatter(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex], self.zValues[pltIndex][key][zIndex], **scatterPlotOptions)
                      if 'colorbar' not in self.options.keys() or self.options['colorbar']['colorbar'] != 'off':
                        if first:
                          m = matplotlib.cm.ScalarMappable(cmap = self.actPlot.cmap, norm = self.actPlot.norm)
                          m.set_array(self.colorMapValues[pltIndex][key])
                          self.actcm = self.fig.colorbar(m)
                          self.actcm.set_label(self.colorMapCoordinates[pltIndex][0].split('|')[-1].replace(')', ''))
                        else:
                          m = matplotlib.cm.ScalarMappable(cmap = self.actPlot.cmap, norm = self.actPlot.norm)
                          m.set_clim(vmin = min(self.colorMapValues[pltIndex][key][-1]), vmax = max(self.colorMapValues[pltIndex][key][-1]))
                          self.actcm.draw_all()
                  else:
                    if 'color' not in scatterPlotOptions:
                      scatterPlotOptions['c'] = plotSettings['c']
                    self.actPlot = self.plt3D.scatter(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex], self.zValues[pltIndex][key][zIndex], **scatterPlotOptions)

      #################
      #   LINE PLOT   #
      #################
      elif self.outStreamTypes[pltIndex] == 'line':
        minV = 0
        maxV = 0
        ## If the user does not define an appropriate cmap, then use matplotlib's default.
        if 'cmap' not in plotSettings or plotSettings['cmap'] not in matplotlib.cm.datad.keys():
          plotSettings['cmap'] = None
        if bool(self.colorMapValues):
          for key in self.xValues[pltIndex].keys():
            minV = min(minV,self.colorMapValues[pltIndex][key][-1][-1])
            maxV = max(maxV,self.colorMapValues[pltIndex][key][-1][-1])
          cmap = matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(minV,maxV,True), plotSettings['cmap'])
          cmap.set_array([minV,maxV])
        for key in self.xValues[pltIndex].keys():
          for xIndex in range(len(self.xValues[pltIndex][key])):
            if self.colorMapCoordinates[pltIndex] != None:
              plotSettings['interpPointsX'] = str(max(200, len(self.xValues[pltIndex][key][xIndex])))
            for yIndex in range(len(self.yValues[pltIndex][key])):
              if self.dim == 2:
                if self.yValues[pltIndex][key][yIndex].size < 2:
                  return
                xi, yi = mathUtils.interpolateFunction(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex], plotSettings, returnCoordinate = True)
                if self.colorMapCoordinates[pltIndex] != None:
                  plt.plot(xi, yi, c = cmap.cmap(self.colorMapValues[pltIndex][key][-1][-1]/(maxV-minV)))
                  if 'colorbar' not in self.options.keys() or self.options['colorbar']['colorbar'] != 'off':
                    if self.actcm is None:
                      self.actcm = self.fig.colorbar(cmap)
                      self.actcm.set_label(self.colorMapCoordinates[pltIndex][0].split('|')[-1].replace(')', ''))
                    else:
                      self.actcm.draw_all()
                else:
                  self.actPlot = plt.plot(xi, yi, **plotSettings.get('attributes', {}))
              elif self.dim == 3:
                for zIndex in range(len(self.zValues[pltIndex][key])):
                  if self.zValues[pltIndex][key][zIndex].size <= 3:
                    return
                  if self.colorMapCoordinates[pltIndex] != None:
                    self.plt3D.plot(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex],self.zValues[pltIndex][key][zIndex],
                                    c = cmap.cmap(self.colorMapValues[pltIndex][key][-1][-1]/(maxV-minV)))
                    if 'colorbar' not in self.options.keys() or self.options['colorbar']['colorbar'] != 'off':
                      if self.actcm is None:
                        self.actcm = self.fig.colorbar(cmap)
                        self.actcm.set_label(self.colorMapCoordinates[pltIndex][0].split('|')[-1].replace(')', ''))
                      else:
                        self.actcm.draw_all()
                  else:
                    self.actPlot = self.plt3D.plot(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex], self.zValues[pltIndex][key][zIndex], **plotSettings.get('attributes', {}))
      ##################
      # HISTOGRAM PLOT #
      ##################
      elif self.outStreamTypes[pltIndex] == 'histogram':
        if 'bins' not in plotSettings.keys():
          if self.dim == 2:
            plotSettings['bins'] = '10'
          else:
            plotSettings['bins'] = '4'

        keys = plotSettings.keys()
        if 'normed' not in keys:
          plotSettings['normed'] = 'False'
        if 'weights' not in keys:
          plotSettings['weights'] = 'None'
        if 'cumulative' not in keys:
          plotSettings['cumulative'] = 'False'
        if 'histtype' not in keys:
          plotSettings['histtype'] = 'bar'
        if 'align' not in keys:
          plotSettings['align'] = 'mid'
        if 'orientation' not in keys:
          plotSettings['orientation'] = 'vertical'
        if 'rwidth' not in keys:
          plotSettings['rwidth'] = 'None'
        if 'log' not in keys:
          plotSettings['log'] = 'None'
        if 'color' not in keys:
          plotSettings['color'] = 'b'
        if 'stacked' not in keys:
          plotSettings['stacked'] = 'None'

        #if self.sourceData[0].type.strip() == 'HistorySet' and self.xCoordinates[0][0].split("|")[1] == "Output":
        if self.sourceData[0].type.strip() == 'HistorySet':
          """
            @MANDD: This 'if' condition has been added in order to allow the user the correctly create an histogram out of an historySet
            If the histogram is created out of the input variables, then the plot has an identical meaning of the one generated by a pointSet
            However, if the histogram is created out of the output variables, then the plot consider only the last value of the array
          """
          data={}
          data['x']=np.empty(0)
          data['y']=np.empty(0)
          for index in range(len(self.outStreamTypes)):
            for key in self.xValues[index].keys():
              data['x'] = np.append(data['x'],self.xValues[index][key][0][-1])
              if self.dim == 3:
                data['y'] = np.append(data['y'],self.yValues[index][key][0][-1])
            del(self.xValues[index])
            self.xValues={}
            self.xValues[index]={}
            self.xValues[index][0]=[]
            self.xValues[index][0].append(copy.deepcopy(data['x']))
            if self.dim == 3:
              del(self.yValues[index])
              self.yValues={}
              self.yValues[index]={}
              self.yValues[index][0]=[]
              self.yValues[index][0].append(copy.deepcopy(data['y']))

        for key in self.xValues[pltIndex].keys():
          for xIndex in range(len(self.xValues[pltIndex][key])):
            try:
              colorss = ast.literal_eval(plotSettings['color'])
            except:
              colorss = plotSettings['color']
            if self.dim == 2:
              plt.hist(self.xValues[pltIndex][key][xIndex], bins = ast.literal_eval(plotSettings['bins']), density = ast.literal_eval(plotSettings['normed']), weights = ast.literal_eval(plotSettings['weights']),
                            cumulative = ast.literal_eval(plotSettings['cumulative']), histtype = plotSettings['histtype'], align = plotSettings['align'],
                            orientation = plotSettings['orientation'], rwidth = ast.literal_eval(plotSettings['rwidth']), log = ast.literal_eval(plotSettings['log']),
                            color = colorss, stacked = ast.literal_eval(plotSettings['stacked']), **plotSettings.get('attributes', {}))
            elif self.dim == 3:
              for yIndex in range(len(self.yValues[pltIndex][key])):
                hist, xedges, yedges = np.histogram2d(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex], bins = ast.literal_eval(plotSettings['bins']))
                elements = (len(xedges) - 1) * (len(yedges) - 1)
                if 'x_offset' in plotSettings.keys():
                  xoffset = float(plotSettings['x_offset'])
                else:
                  xoffset = 0.25
                if 'y_offset' in plotSettings.keys():
                  yoffset = float(plotSettings['y_offset'])
                else:
                  yoffset = 0.25
                if 'dx' in plotSettings.keys():
                  dxs = float(plotSettings['dx'])
                else:
                  dxs = (self.xValues[pltIndex][key][xIndex].max() - self.xValues[pltIndex][key][xIndex].min()) / float(plotSettings['bins'])
                if 'dy' in plotSettings.keys():
                  dys = float(plotSettings['dy'])
                else:
                  dys = (self.yValues[pltIndex][key][yIndex].max() - self.yValues[pltIndex][key][yIndex].min()) / float(plotSettings['bins'])
                xpos, ypos = np.meshgrid(xedges[:-1] + xoffset, yedges[:-1] + yoffset)
                self.actPlot = self.plt3D.bar3d(xpos.flatten(), ypos.flatten(), np.zeros(elements), dxs * np.ones_like(elements), dys * np.ones_like(elements), hist.flatten(), color = colorss, zsort = 'average', **plotSettings.get('attributes', {}))
      ##################
      #    STEM PLOT   #
      ##################
      elif self.outStreamTypes[pltIndex] == 'stem':
        if 'linefmt' not in plotSettings.keys():
          plotSettings['linefmt'] = 'b-'
        if 'markerfmt' not in plotSettings.keys():
          plotSettings['markerfmt'] = 'bo'
        if 'basefmt' not in plotSettings.keys():
          plotSettings['basefmt'] = 'r-'
        for key in self.xValues[pltIndex].keys():
          for xIndex in range(len(self.xValues[pltIndex][key])):
            for yIndex in range(len(self.yValues[pltIndex][key])):
              if self.dim == 2:
                self.actPlot = plt.stem(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex], linefmt = plotSettings['linefmt'],
                                        markerfmt = plotSettings['markerfmt'], basefmt = plotSettings['linefmt'],
                                        use_line_collection=True, **plotSettings.get('attributes', {}))
              elif self.dim == 3:
                # it is a basic stem plot constructed using a standard line plot. For now we do not use the previous defined keywords...
                for zIndex in range(len(self.zValues[pltIndex][key])):
                  for xx, yy, zz in zip(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex], self.zValues[pltIndex][key][zIndex]):
                    self.plt3D.plot([xx, xx], [yy, yy], [0, zz], '-')
      ##################
      #    STEP PLOT   #
      ##################
      elif self.outStreamTypes[pltIndex] == 'step':
        if self.dim == 2:
          if 'where' not in plotSettings.keys():
            plotSettings['where'] = 'mid'
          for key in self.xValues[pltIndex].keys():
            for xIndex in range(len(self.xValues[pltIndex][key])):
              if self.xValues[pltIndex][key][xIndex].size < 2:
                xi = self.xValues[pltIndex][key][xIndex]
              else:
                xi = np.linspace(self.xValues[pltIndex][key][xIndex].min(), self.xValues[pltIndex][key][xIndex].max(), ast.literal_eval(plotSettings['interpPointsX']))
              for yIndex in range(len(self.yValues[pltIndex][key])):
                if self.yValues[pltIndex][key][yIndex].size <= 3:
                  return
                yi = mathUtils.interpolateFunction(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex], plotSettings)
                self.actPlot = plt.step(xi, yi, where = plotSettings['where'], **plotSettings.get('attributes', {}))
        elif self.dim == 3:
          self.raiseAWarning('step Plot not available in 3D')
          return
      ########################
      #    PSEUDOCOLOR PLOT  #
      ########################
      elif self.outStreamTypes[pltIndex] == 'pseudocolor':
        if self.dim == 2:
          for key in self.xValues[pltIndex].keys():
            for xIndex in range(len(self.xValues[pltIndex][key])):
              ## Hopefully, x,y, and z are all the same length, so checking this
              ## here should be good enough.
              ## The problem is you cannot interpolate any amount of space if
              ## you only have a single data point.
              if self.xValues[pltIndex][key][xIndex].size == 1:
                self.raiseAWarning('Nothing to Plot Yet. Continuing to next plot.')
                continue
              for yIndex in range(len(self.yValues[pltIndex][key])):
                if not self.colorMapCoordinates:
                  self.raiseAMessage('pseudocolor Plot needs coordinates for color map... Returning without plotting')
                  return
                for zIndex in range(len(self.colorMapValues[pltIndex][key])):
                  if self.colorMapValues[pltIndex][key][zIndex].size <= 3:
                    return
                  xig, yig, Ci = mathUtils.interpolateFunction(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex], plotSettings, z = self.colorMapValues[pltIndex][key][zIndex], returnCoordinate = True)
                  if plotSettings['cmap'] == 'None':
                    self.actPlot = plt.pcolormesh(xig, yig, ma.masked_where(np.isnan(Ci), Ci), **plotSettings.get('attributes', {}))
                    m = matplotlib.cm.ScalarMappable(cmap = self.actPlot.cmap, norm = self.actPlot.norm)
                  else:
                    self.actPlot = plt.pcolormesh(xig, yig, ma.masked_where(np.isnan(Ci), Ci), cmap = matplotlib.cm.get_cmap(name = plotSettings['cmap']), **plotSettings.get('attributes', {}))
                    m = matplotlib.cm.ScalarMappable(cmap = self.actPlot.cmap, norm = self.actPlot.norm)
                  m.set_array(ma.masked_where(np.isnan(Ci), Ci))
                  if 'colorbar' not in self.options.keys() or self.options['colorbar']['colorbar'] != 'off':
                    actcm = self.fig.colorbar(m)
                    actcm.set_label(self.colorMapCoordinates[pltIndex][0].split('|')[-1].replace(')', ''))
        elif self.dim == 3:
          self.raiseAWarning('pseudocolor Plot is considered a 2D plot, not a 3D!')
          return
      ########################
      #     SURFACE PLOT     #
      ########################
      elif self.outStreamTypes[pltIndex] == 'surface':
        if self.dim == 2:
          self.raiseAWarning('surface Plot is NOT available for 2D plots, IT IS A 3D!')
          return
        elif self.dim == 3:
          if 'rstride' not in plotSettings.keys():
            plotSettings['rstride'] = '1'
          if 'cstride' not in plotSettings.keys():
            plotSettings['cstride'] = '1'
          if 'antialiased' not in plotSettings.keys():
            plotSettings['antialiased'] = 'False'
          if 'linewidth' not in plotSettings.keys():
            plotSettings['linewidth'] = '0'
          for key in self.xValues[pltIndex].keys():
            for xIndex in range(len(self.xValues[pltIndex][key])):
              ## Hopefully, x,y, and z are all the same length, so checking this
              ## here should be good enough.
              ## The problem is you cannot interpolate any amount of space if
              ## you only have a single data point.
              if self.xValues[pltIndex][key][xIndex].size == 1:
                self.raiseAWarning('Nothing to Plot Yet. Continuing to next plot.')
                continue
              for yIndex in range(len(self.yValues[pltIndex][key])):
                for zIndex in range(len(self.zValues[pltIndex][key])):
                  if self.zValues[pltIndex][key][zIndex].size <= 3:
                    return
                  if self.colorMapCoordinates[pltIndex] != None:
                    xig, yig, Ci = mathUtils.interpolateFunction(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex], plotSettings, z = self.colorMapValues[pltIndex][key][zIndex], returnCoordinate = True)
                  xig, yig, zi = mathUtils.interpolateFunction(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex], plotSettings, z = self.zValues[pltIndex][key][zIndex], returnCoordinate = True)
                  if self.colorMapCoordinates[pltIndex] != None:
                    if self.actcm:
                      first = False
                    else:
                      first = True
                    if plotSettings['cmap'] == 'None':
                      plotSettings['cmap'] = 'jet'
                    self.actPlot = self.plt3D.plot_surface(xig, yig, ma.masked_where(np.isnan(zi), zi), rstride = ast.literal_eval(plotSettings['rstride']), cstride = ast.literal_eval(plotSettings['cstride']), facecolors = matplotlib.cm.get_cmap(name = plotSettings['cmap'])(ma.masked_where(np.isnan(Ci), Ci)), cmap = matplotlib.cm.get_cmap(name = plotSettings['cmap']), linewidth = ast.literal_eval(plotSettings['linewidth']), antialiased = ast.literal_eval(plotSettings['antialiased']), **plotSettings.get('attributes', {}))
                    if first:
                      self.actPlot.cmap = matplotlib.cm.get_cmap(name = plotSettings['cmap'])
                    if 'colorbar' not in self.options.keys() or self.options['colorbar']['colorbar'] != 'off':
                      if first:
                        m = matplotlib.cm.ScalarMappable(cmap = self.actPlot.cmap, norm = self.actPlot.norm)
                        m.set_array(self.colorMapValues[pltIndex][key])
                        self.actcm = self.fig.colorbar(m)
                        self.actcm.set_label(self.colorMapCoordinates[pltIndex][0].split('|')[-1].replace(')', ''))
                      else:
                        m = matplotlib.cm.ScalarMappable(cmap = self.actPlot.cmap, norm = self.actPlot.norm)
                        m.set_clim(vmin = min(self.colorMapValues[pltIndex][key][-1]), vmax = max(self.colorMapValues[pltIndex][key][-1]))
                        self.actcm.draw_all()
                  else:
                    if plotSettings['cmap'] == 'None':
                      self.actPlot = self.plt3D.plot_surface(xig, yig, ma.masked_where(np.isnan(zi), zi), rstride = ast.literal_eval(plotSettings['rstride']), cstride = ast.literal_eval(plotSettings['cstride']), linewidth = ast.literal_eval(plotSettings['linewidth']), antialiased = ast.literal_eval(plotSettings['antialiased']), **plotSettings.get('attributes', {}))
                      if 'color' in plotSettings.get('attributes', {}).keys():
                        self.actPlot.set_color = plotSettings.get('attributes', {})['color']
                      else:
                        self.actPlot.set_color = 'blue'
                    else:
                      self.actPlot = self.plt3D.plot_surface(xig, yig, ma.masked_where(np.isnan(zi), zi), rstride = ast.literal_eval(plotSettings['rstride']), cstride = ast.literal_eval(plotSettings['cstride']), cmap = matplotlib.cm.get_cmap(name = plotSettings['cmap']), linewidth = ast.literal_eval(plotSettings['linewidth']), antialiased = ast.literal_eval(plotSettings['antialiased']), **plotSettings.get('attributes', {}))
      ########################
      #   TRI-SURFACE PLOT   #
      ########################
      elif self.outStreamTypes[pltIndex] == 'tri-surface':
        if self.dim == 2:
          self.raiseAWarning('TRI-surface Plot is NOT available for 2D plots, it is 3D!')
          return
        elif self.dim == 3:
          if 'color' not in plotSettings.keys():
            plotSettings['color'] = 'b'
          if 'shade' not in plotSettings.keys():
            plotSettings['shade'] = 'False'
          for key in self.xValues[pltIndex].keys():
            for xIndex in range(len(self.xValues[pltIndex][key])):
              ## Hopefully, x,y, and z are all the same length, so checking this
              ## here should be good enough.
              ## The problem is you cannot interpolate any amount of space if
              ## you only have a single data point.
              if self.xValues[pltIndex][key][xIndex].size == 1:
                self.raiseAWarning('Nothing to Plot Yet. Continuing to next plot.')
                continue
              for yIndex in range(len(self.yValues[pltIndex][key])):
                for zIndex in range(len(self.zValues[pltIndex][key])):
                  metric = (self.xValues[pltIndex][key][xIndex] ** 2 + self.yValues[pltIndex][key][yIndex] ** 2) ** 0.5
                  metricIndeces = np.argsort(metric)
                  xs = np.zeros(self.xValues[pltIndex][key][xIndex].shape)
                  ys = np.zeros(self.yValues[pltIndex][key][yIndex].shape)
                  zs = np.zeros(self.zValues[pltIndex][key][zIndex].shape)
                  for sindex in range(len(metricIndeces)):
                    xs[sindex] = self.xValues[pltIndex][key][xIndex][metricIndeces[sindex]]
                    ys[sindex] = self.yValues[pltIndex][key][yIndex][metricIndeces[sindex]]
                    zs[sindex] = self.zValues[pltIndex][key][zIndex][metricIndeces[sindex]]
                  surfacePlotOptions = {'color': plotSettings['color'],
                                        'shade':ast.literal_eval(plotSettings['shade'])}
                  surfacePlotOptions.update(plotSettings.get('attributes', {}))
                  if self.zValues[pltIndex][key][zIndex].size <= 3:
                    return
                  if self.colorMapCoordinates[pltIndex] != None:
                    if self.actcm:
                      first = False
                    else:
                      first = True
                    if plotSettings['cmap'] == 'None':
                      plotSettings['cmap'] = 'jet'
                    surfacePlotOptions['cmap'] = matplotlib.cm.get_cmap(name = plotSettings['cmap'])
                    self.actPlot = self.plt3D.plot_trisurf(xs, ys, zs, **surfacePlotOptions)
                    if 'colorbar' not in self.options.keys() or self.options['colorbar']['colorbar'] != 'off':
                      if first:
                        self.actPlot.cmap = matplotlib.cm.get_cmap(name = plotSettings['cmap'])
                        m = matplotlib.cm.ScalarMappable(cmap = self.actPlot.cmap, norm = self.actPlot.norm)
                        m.set_array(self.colorMapValues[pltIndex][key])
                        self.actcm = self.fig.colorbar(m)
                        self.actcm.set_label(self.colorMapCoordinates[pltIndex][0].split('|')[-1].replace(')', ''))
                      else:
                        m = matplotlib.cm.ScalarMappable(cmap = self.actPlot.cmap, norm = self.actPlot.norm)
                        m.set_clim(vmin = min(self.colorMapValues[pltIndex][key][-1]), vmax = max(self.colorMapValues[pltIndex][key][-1]))
                        self.actcm.draw_all()
                  else:
                    if plotSettings['cmap'] != 'None':
                      surfacePlotOptions["cmap"] = matplotlib.cm.get_cmap(name = plotSettings['cmap'])
                    self.actPlot = self.plt3D.plot_trisurf(xs, ys, zs, **surfacePlotOptions)
      ########################
      #    WIREFRAME  PLOT   #
      ########################
      elif self.outStreamTypes[pltIndex] == 'wireframe':
        if self.dim == 2:
          self.raiseAWarning('wireframe Plot is NOT available for 2D plots, IT IS A 3D!')
          return
        elif self.dim == 3:
          if 'rstride' not in plotSettings.keys():
            plotSettings['rstride'] = '1'
          if 'cstride' not in plotSettings.keys():
            plotSettings['cstride'] = '1'
          for key in self.xValues[pltIndex].keys():
            for xIndex in range(len(self.xValues[pltIndex][key])):
              ## Hopefully, x,y, and z are all the same length, so checking this
              ## here should be good enough.
              ## The problem is you cannot interpolate any amount of space if
              ## you only have a single data point.
              if self.xValues[pltIndex][key][xIndex].size == 1:
                self.raiseAWarning('Nothing to Plot Yet. Continuing to next plot.')
                continue
              for yIndex in range(len(self.yValues[pltIndex][key])):
                for zIndex in range(len(self.zValues[pltIndex][key])):
                  if self.zValues[pltIndex][key][zIndex].size <= 3:
                    return
                  if self.colorMapCoordinates[pltIndex] != None:
                    xig, yig, Ci = mathUtils.interpolateFunction(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex], plotSettings, z = self.colorMapValues[pltIndex][key][zIndex], returnCoordinate = True)
                  xig, yig, zi = mathUtils.interpolateFunction(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex], plotSettings, z = self.zValues[pltIndex][key][zIndex], returnCoordinate = True)
                  if self.colorMapCoordinates[pltIndex] != None:
                    self.raiseAWarning('Currently, ax.plot_wireframe() in MatPlotLib version: ' + matplotlib.__version__ + ' does not support a colormap! Wireframe plotted on a surface plot...')
                    if self.actcm:
                      first = False
                    else:
                      first = True
                    if plotSettings['cmap'] == 'None':
                      plotSettings['cmap'] = 'jet'
                    self.actPlot = self.plt3D.plot_wireframe(xig, yig, ma.masked_where(np.isnan(zi), zi), rstride = ast.literal_eval(plotSettings['rstride']), cmap = matplotlib.cm.get_cmap(name = plotSettings['cmap']), cstride = ast.literal_eval(plotSettings['cstride']), **plotSettings.get('attributes', {}))
                    self.actPlot = self.plt3D.plot_surface(xig, yig, ma.masked_where(np.isnan(zi), zi), alpha = 0.4, rstride = ast.literal_eval(plotSettings['rstride']), cmap = matplotlib.cm.get_cmap(name = plotSettings['cmap']), cstride = ast.literal_eval(plotSettings['cstride']), **plotSettings.get('attributes', {}))
                    if 'colorbar' not in self.options.keys() or self.options['colorbar']['colorbar'] != 'off':
                      if first:
                        m = matplotlib.cm.ScalarMappable(cmap = self.actPlot.cmap, norm = self.actPlot.norm)
                        m.set_array(self.colorMapValues[pltIndex][key])
                        self.actcm = self.fig.colorbar(m)
                        self.actcm.set_label(self.colorMapCoordinates[pltIndex][0].split('|')[-1].replace(')', ''))
                      else:
                        m = matplotlib.cm.ScalarMappable(cmap = self.actPlot.cmap, norm = self.actPlot.norm)
                        m.set_clim(vmin = min(self.colorMapValues[pltIndex][key][-1]), vmax = max(self.colorMapValues[pltIndex][key][-1]))
                        self.actcm.draw_all()
                  else:
                    if plotSettings['cmap'] == 'None':
                      self.actPlot = self.plt3D.plot_wireframe(xig, yig, ma.masked_where(np.isnan(zi), zi), rstride = ast.literal_eval(plotSettings['rstride']), cstride = ast.literal_eval(plotSettings['cstride']), **plotSettings.get('attributes', {}))
                      if 'color' in plotSettings.get('attributes', {}).keys():
                        self.actPlot.set_color = plotSettings.get('attributes', {})['color']
                      else:
                        self.actPlot.set_color = 'blue'
                    else:
                      self.actPlot = self.plt3D.plot_wireframe(xig, yig, ma.masked_where(np.isnan(zi), zi), rstride = ast.literal_eval(plotSettings['rstride']), cstride = ast.literal_eval(plotSettings['cstride']), **plotSettings.get('attributes', {}))

      ########################
      #     CONTOUR   PLOT   #
      ########################
      elif self.outStreamTypes[pltIndex] == 'contour' or self.outStreamTypes[pltIndex] == 'filledContour':
        if self.dim == 2:
          if 'numberBins' in plotSettings.keys():
            nbins = int(plotSettings['numberBins'])
          else:
            nbins = 5
          for key in self.xValues[pltIndex].keys():
            if not self.colorMapCoordinates:
              self.raiseAWarning(self.outStreamTypes[pltIndex] + ' Plot needs coordinates for color map... Returning without plotting')
              return
            for xIndex in range(len(self.xValues[pltIndex][key])):
              ## Hopefully, x,y, and z are all the same length, so checking this
              ## here should be good enough.
              ## The problem is you cannot interpolate any amount of space if
              ## you only have a single data point.
              if self.xValues[pltIndex][key][xIndex].size == 1:
                self.raiseAWarning('Nothing to Plot Yet. Continuing to next plot.')
                continue
              for yIndex in range(len(self.yValues[pltIndex][key])):
                for zIndex in range(len(self.colorMapValues[pltIndex][key])):
                  if self.actcm:
                    first = False
                  else:
                    first = True
                  xig, yig, Ci = mathUtils.interpolateFunction(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex], plotSettings, z = self.colorMapValues[pltIndex][key][zIndex], returnCoordinate = True)
                  if self.outStreamTypes[pltIndex] == 'contour':
                    if plotSettings['cmap'] == 'None':
                      if 'color' in plotSettings.get('attributes', {}).keys():
                        color = plotSettings.get('attributes', {})['color']
                      else:
                        color = 'blue'
                      self.actPlot = plt.contour(xig, yig, ma.masked_where(np.isnan(Ci), Ci), nbins, colors = color, **plotSettings.get('attributes', {}))
                    else:
                      self.actPlot = plt.contour(xig, yig, ma.masked_where(np.isnan(Ci), Ci), nbins, **plotSettings.get('attributes', {}))
                  else:
                    if plotSettings['cmap'] == 'None':
                      plotSettings['cmap'] = 'jet'
                    self.actPlot = plt.contourf(xig, yig, ma.masked_where(np.isnan(Ci), Ci), nbins, **plotSettings.get('attributes', {}))
                  plt.clabel(self.actPlot, inline = 1, fontsize = 10)
                  if 'colorbar' not in self.options.keys() or self.options['colorbar']['colorbar'] != 'off':
                    if first:
                      self.actcm = plt.colorbar(self.actPlot, shrink = 0.8, extend = 'both')
                      self.actcm.set_label(self.colorMapCoordinates[pltIndex][0].split('|')[-1].replace(')', ''))
                    else:
                      m = matplotlib.cm.ScalarMappable(cmap = self.actPlot.cmap, norm = self.actPlot.norm)
                      m.set_clim(vmin = min(self.colorMapValues[pltIndex][key][-1]), vmax = max(self.colorMapValues[pltIndex][key][-1]))
                      self.actcm.draw_all()
        elif self.dim == 3:
          self.raiseAWarning('contour/filledContour is a 2-D plot, where x,y are the surface coordinates and colorMap vector is the array to visualize!\n               contour3D/filledContour3D are 3-D! ')
          return
      ## These should be combined: ^^^ & vvv
      elif self.outStreamTypes[pltIndex] == 'contour3D' or self.outStreamTypes[pltIndex] == 'filledContour3D':
        if self.dim == 2:
          self.raiseAWarning('contour3D/filledContour3D Plot is NOT available for 2D plots, IT IS A 2D! Check "contour/filledContour"!')
          return
        elif self.dim == 3:
          if 'numberBins' in plotSettings.keys():
            nbins = int(plotSettings['numberBins'])
          else:
            nbins = 5
          if 'extend3D' in plotSettings.keys():
            ext3D = bool(plotSettings['extend3D'])
          else:
            ext3D = False
          for key in self.xValues[pltIndex].keys():
            for xIndex in range(len(self.xValues[pltIndex][key])):
              ## Hopefully, x,y, and z are all the same length, so checking this
              ## here should be good enough.
              ## The problem is you cannot interpolate any amount of space if
              ## you only have a single data point.
              if self.xValues[pltIndex][key][xIndex].size == 1:
                self.raiseAWarning('Nothing to Plot Yet. Continuing to next plot.')
                continue
              for yIndex in range(len(self.yValues[pltIndex][key])):
                for zIndex in range(len(self.colorMapValues[pltIndex][key])):
                  if self.actcm:
                    first = False
                  else:
                    first = True
                  xig, yig, Ci = mathUtils.interpolateFunction(self.xValues[pltIndex][key][xIndex], self.yValues[pltIndex][key][yIndex], plotSettings, z = self.colorMapValues[pltIndex][key][zIndex], returnCoordinate = True)
                  if self.outStreamTypes[pltIndex] == 'contour3D':
                    if plotSettings['cmap'] == 'None':
                      if 'color' in plotSettings.get('attributes', {}).keys():
                        color = plotSettings.get('attributes', {})['color']
                      else:
                        color = 'blue'
                      self.actPlot = self.plt3D.contour3D(xig, yig, ma.masked_where(np.isnan(Ci), Ci), nbins, colors = color, extend3d = ext3D, **plotSettings.get('attributes', {}))
                    else:
                      self.actPlot = self.plt3D.contour3D(xig, yig, ma.masked_where(np.isnan(Ci), Ci), nbins, extend3d = ext3D, cmap = matplotlib.cm.get_cmap(name = plotSettings['cmap']), **plotSettings.get('attributes', {}))
                  else:
                    if plotSettings['cmap'] == 'None':
                      plotSettings['cmap'] = 'jet'
                    self.actPlot = self.plt3D.contourf3D(xig, yig, ma.masked_where(np.isnan(Ci), Ci), nbins, cmap = matplotlib.cm.get_cmap(name = plotSettings['cmap']), **plotSettings.get('attributes', {}))
                  plt.clabel(self.actPlot, inline = 1, fontsize = 10)
                  if 'colorbar' not in self.options.keys() or self.options['colorbar']['colorbar'] != 'off':
                    if first:
                      self.actcm = plt.colorbar(self.actPlot, shrink = 0.8, extend = 'both')
                      self.actcm.set_label(self.colorMapCoordinates[pltIndex][0].split('|')[-1].replace(')', ''))
                    else:
                      m = matplotlib.cm.ScalarMappable(cmap = self.actPlot.cmap, norm = self.actPlot.norm)
                      m.set_clim(vmin = min(self.colorMapValues[pltIndex][key][-1]), vmax = max(self.colorMapValues[pltIndex][key][-1]))
                      self.actcm.draw_all()
      ########################
      #   DataMining PLOT    #
      ########################
      elif self.outStreamTypes[pltIndex] == 'dataMining':
        from itertools import cycle
        from itertools import cycle
        colors = cycle(['#88CCEE', '#DDCC77', '#AA4499', '#117733', '#332288', '#999933', '#44AA99', '#882255', '#CC6677', '#CD6677', '#DC6877', '#886677', '#AA6677', '#556677', '#CD7865'])
        if 's' not in plotSettings.keys():
          plotSettings['s'] = '20'
        if 'c' not in plotSettings.keys():
          plotSettings['c'] = 'b'
        if 'marker' not in plotSettings.keys():
          plotSettings['marker'] = 'o'
        if 'alpha' not in plotSettings.keys():
          plotSettings['alpha'] = 'None'
        if 'linewidths' not in plotSettings.keys():
          plotSettings['linewidths'] = 'None'
        clusterDict[pltIndex] = {}
        for key in self.xValues[pltIndex].keys():
          for xIndex in range(len(self.xValues[pltIndex][key])):
            for yIndex in range(len(self.yValues[pltIndex][key])):
              dataMiningPlotOptions = {'s':ast.literal_eval(plotSettings['s']),
                                       'marker':(plotSettings['marker']),
                                       'alpha':ast.literal_eval(plotSettings['alpha']),
                                       'linewidths':ast.literal_eval(plotSettings['linewidths'])}
              if self.colorMapCoordinates[pltIndex] != None:
                self.raiseAWarning('ColorMap values supplied, however DataMining plots do not use colorMap from input.')
              if plotSettings['cmap'] == 'None':
                self.raiseAWarning('ColorSet supplied, however DataMining plots do not use color set from input.')
              if 'cluster' == plotSettings['SKLtype']:
                # TODO: include the cluster Centers to the plot
                if 'noClusters' in plotSettings.get('attributes', {}).keys():
                  clusterDict[pltIndex]['noClusters'] = int(plotSettings.get('attributes', {})['noClusters'])
                  plotSettings.get('attributes', {}).pop('noClusters')
                else:
                  clusterDict[pltIndex]['noClusters'] = np.amax(self.clusterValues[pltIndex][1][0]) + 1
                dataMiningPlotOptions.update(plotSettings.get('attributes', {}))
                if   self.dim == 2:
                  clusterDict[pltIndex]['clusterValues'] = np.zeros(shape = (len(self.xValues[pltIndex][key][xIndex]), 2))
                elif self.dim == 3:
                  clusterDict[pltIndex]['clusterValues'] = np.zeros(shape = (len(self.xValues[pltIndex][key][xIndex]), 3))
                clusterDict[pltIndex]['clusterValues'][:, 0] = self.xValues[pltIndex][key][xIndex]
                clusterDict[pltIndex]['clusterValues'][:, 1] = self.yValues[pltIndex][key][yIndex]
                if self.dim == 2:
                  for k, col in zip(range(int(clusterDict[pltIndex]['noClusters'])), colors):
                    myMembers = self.clusterValues[pltIndex][1][0] == k
                    self.actPlot = plt.scatter(clusterDict[pltIndex]['clusterValues'][myMembers, 0], clusterDict[pltIndex]['clusterValues'][myMembers, 1] , color = col, **dataMiningPlotOptions)

                  ## Handle all of the outlying data
                  myMembers = self.clusterValues[pltIndex][1][0] == -1
                  ## resize the points
                  dataMiningPlotOptions['s'] /= 2
                  ## and hollow out their markers
                  if 'facecolors' in dataMiningPlotOptions:
                    faceColors = dataMiningPlotOptions['facecolors']
                  else:
                    faceColors = None
                  dataMiningPlotOptions['facecolors'] = 'none'

                  self.actPlot = plt.scatter(clusterDict[pltIndex]['clusterValues'][myMembers, 0], clusterDict[pltIndex]['clusterValues'][myMembers, 1] , color = '#000000', **dataMiningPlotOptions)

                  ## Restore the plot options to their original values
                  dataMiningPlotOptions['s'] *= 2
                  if faceColors is not None:
                    dataMiningPlotOptions['facecolors'] = faceColors
                  else:
                    del dataMiningPlotOptions['facecolors']

                elif self.dim == 3:
                  for zIndex in range(len(self.zValues[pltIndex][key])):
                    clusterDict[pltIndex]['clusterValues'][:, 2] = self.zValues[pltIndex][key][zIndex]
                  for k, col in zip(range(int(clusterDict[pltIndex]['noClusters'])), colors):
                    myMembers = self.clusterValues[pltIndex][1][0] == k
                    self.actPlot = self.plt3D.scatter(clusterDict[pltIndex]['clusterValues'][myMembers, 0], clusterDict[pltIndex]['clusterValues'][myMembers, 1], clusterDict[pltIndex]['clusterValues'][myMembers, 2], color = col, **dataMiningPlotOptions)

                  ## Handle all of the outlying data
                  myMembers = self.clusterValues[pltIndex][1][0] == -1
                  ## resize the points
                  dataMiningPlotOptions['s'] /= 2
                  ## and hollow out their markers
                  if 'facecolors' in dataMiningPlotOptions:
                    faceColors = dataMiningPlotOptions['facecolors']
                  else:
                    faceColors = None
                  dataMiningPlotOptions['facecolors'] = 'none'

                  self.actPlot = self.plt3D.scatter(clusterDict[pltIndex]['clusterValues'][myMembers, 0], clusterDict[pltIndex]['clusterValues'][myMembers, 1], clusterDict[pltIndex]['clusterValues'][myMembers, 2], color = '#000000', **dataMiningPlotOptions)

                  ## Restore the plot options to their original values
                  dataMiningPlotOptions['s'] *= 2
                  if faceColors is not None:
                    dataMiningPlotOptions['facecolors'] = faceColors
                  else:
                    del dataMiningPlotOptions['facecolors']

              elif 'bicluster' == plotSettings['SKLtype']:
                self.raiseAnError(IOError, 'SKLType Bi-Cluster Plots are not implemented yet!..')
              elif 'mixture' == plotSettings['SKLtype']:
                if 'noMixtures' in plotSettings.get('attributes', {}).keys():
                  clusterDict[pltIndex]['noMixtures'] = int(plotSettings.get('attributes', {})['noMixtures'])
                  plotSettings.get('attributes', {}).pop('noMixtures')
                else:
                  clusterDict[pltIndex]['noMixtures'] = np.amax(self.mixtureValues[pltIndex][1][0]) + 1
                if self.dim == 3:
                  self.raiseAnError(IOError, 'SKLType Mixture Plots are only available in 2-Dimensions')
                else:
                  clusterDict[pltIndex]['mixtureValues'] = np.zeros(shape = (len(self.xValues[pltIndex][key][xIndex]), 2))
                clusterDict[pltIndex]['mixtureValues'][:, 0] = self.xValues[pltIndex][key][xIndex]
                clusterDict[pltIndex]['mixtureValues'][:, 1] = self.yValues[pltIndex][key][yIndex]
                if 'mixtureCovars' in plotSettings.get('attributes', {}).keys():
                  split = self.__splitVariableNames('mixtureCovars', (pltIndex, 0))
                  mixtureCovars = self.sourceData[pltIndex].getParam(split[1], split[2], nodeId = 'ending')
                  plotSettings.get('attributes', {}).pop('mixtureCovars')
                else:
                  mixtureCovars = None
                if 'mixtureMeans' in plotSettings.get('attributes', {}).keys():
                  split = self.__splitVariableNames('mixtureMeans', (pltIndex, 0))
                  mixtureMeans = self.sourceData[pltIndex].getParam(split[1], split[2], nodeId = 'ending')
                  plotSettings.get('attributes', {}).pop('mixtureMeans')
                else:
                  mixtureMeans = None
                # mixtureCovars.reshape(3, 4)
                # mixtureMeans.reshape(3, 4)
                # for i, (mean, covar, col) in enumerate(zip(mixtureMeans, mixtureCovars, colors)):
                for i, col in zip(range(clusterDict[pltIndex]['noMixtures']), colors):
                  if not np.any(self.mixtureValues[pltIndex][1][0] == i):
                    continue
                  myMembers = self.mixtureValues[pltIndex][1][0] == i
                  # self.make_ellipses(mixtureCovars, mixtureMeans, i, col)
                  self.actPlot = plt.scatter(clusterDict[pltIndex]['mixtureValues'][myMembers, 0], clusterDict[pltIndex]['mixtureValues'][myMembers, 1], color = col, **dataMiningPlotOptions)
              elif 'manifold' == plotSettings['SKLtype']:
                if   self.dim == 2:
                  manifoldValues = np.zeros(shape = (len(self.xValues[pltIndex][key][xIndex]), 2))
                elif self.dim == 3:
                  manifoldValues = np.zeros(shape = (len(self.xValues[pltIndex][key][xIndex]), 3))
                manifoldValues[:, 0] = self.xValues[pltIndex][key][xIndex]
                manifoldValues[:, 1] = self.yValues[pltIndex][key][yIndex]
                if 'clusterLabels' in plotSettings.get('attributes', {}).keys():
                  split = self.__splitVariableNames('clusterLabels', (pltIndex, 0))
                  clusterDict[pltIndex]['clusterLabels'] = self.sourceData[pltIndex].getParam(split[1], split[2], nodeId = 'ending')
                  plotSettings.get('attributes', {}).pop('clusterLabels')
                else:
                  clusterDict[pltIndex]['clusterLabels'] = None
                if 'noClusters' in plotSettings.get('attributes', {}).keys():
                  clusterDict[pltIndex]['noClusters'] = int(plotSettings.get('attributes', {})['noClusters'])
                  plotSettings.get('attributes', {}).pop('noClusters')
                else:
                  clusterDict[pltIndex]['noClusters'] = np.amax(self.clusterValues[pltIndex][1][0]) + 1
                if self.clusterValues[pltIndex][1][0] is not None:
                  if   self.dim == 2:
                    for k, col in zip(range(clusterDict[pltIndex]['noClusters']), colors):
                      myMembers = self.clusterValues[pltIndex][1][0] == k
                      self.actPlot = plt.scatter(manifoldValues[myMembers, 0], manifoldValues[myMembers, 1], color = col, **dataMiningPlotOptions)
                  elif self.dim == 3:
                    for zIndex in range(len(self.zValues[pltIndex][key])):
                      manifoldValues[:, 2] = self.zValues[pltIndex][key][zIndex]
                    for k, col in zip(range(clusterDict[pltIndex]['noClusters']), colors):
                      myMembers = self.clusterValues[pltIndex][1][0] == k
                      self.actPlot = self.plt3D.scatter(manifoldValues[myMembers, 0], manifoldValues[myMembers, 1], manifoldValues[myMembers, 2], color = col, **dataMiningPlotOptions)
                else:
                  if   self.dim == 2:
                    self.actPlot = plt.scatter(manifoldValues[:, 0], manifoldValues[:, 1], **dataMiningPlotOptions)
                  elif self.dim == 3:
                    for zIndex in range(len(self.zValues[pltIndex][key])):
                      manifoldValues[:, 2] = self.zValues[pltIndex][key][zIndex]
                      self.actPlot = self.plt3D.scatter(manifoldValues[:, 0], manifoldValues[:, 1], manifoldValues[:, 2], **dataMiningPlotOptions)
              elif 'decomposition' == plotSettings['SKLtype']:
                if   self.dim == 2:
                  decompositionValues = np.zeros(shape = (len(self.xValues[pltIndex][key][xIndex]), 2))
                elif self.dim == 3:
                  decompositionValues = np.zeros(shape = (len(self.xValues[pltIndex][key][xIndex]), 3))
                decompositionValues[:, 0] = self.xValues[pltIndex][key][xIndex]
                decompositionValues[:, 1] = self.yValues[pltIndex][key][yIndex]
                if 'noClusters' in plotSettings.get('attributes', {}).keys():
                  clusterDict[pltIndex]['noClusters'] = int(plotSettings.get('attributes', {})['noClusters'])
                  plotSettings.get('attributes', {}).pop('noClusters')
                else:
                  clusterDict[pltIndex]['noClusters'] = np.amax(self.clusterValues[pltIndex][1][0]) + 1
                if self.clusterValues[pltIndex][1][0] is not None:
                  if self.dim == 2:
                    for k, col in zip(range(clusterDict[pltIndex]['noClusters']), colors):
                      myMembers = self.clusterValues[pltIndex][1][0] == k
                      self.actPlot = plt.scatter(decompositionValues[myMembers, 0], decompositionValues[myMembers, 1], color = col, **dataMiningPlotOptions)
                  elif self.dim == 3:
                    for zIndex in range(len(self.zValues[pltIndex][key])):
                      decompositionValues[:, 2] = self.zValues[pltIndex][key][zIndex]
                    for k, col in zip(range(clusterDict[pltIndex]['noClusters']), colors):
                      myMembers = self.clusterValues[pltIndex][1][0] == k
                      self.actPlot = self.plt3D.scatter(decompositionValues[myMembers, 0], decompositionValues[myMembers, 1], decompositionValues[myMembers, 2], color = col, **dataMiningPlotOptions)
                else:
                  # no ClusterLabels
                  if self.dim == 2:
                    self.actPlot = plt.scatter(decompositionValues[:, 0], decompositionValues[:, 1], **dataMiningPlotOptions)
                  elif self.dim == 3:
                    for zIndex in range(len(self.zValues[pltIndex][key])):
                      decompositionValues[:, 2] = self.zValues[pltIndex][key][zIndex]
                      self.actPlot = self.plt3D.scatter(decompositionValues[:, 0], decompositionValues[:, 1], decompositionValues[:, 2], **dataMiningPlotOptions)
      else:
        # Let's try to "write" the code for the plot on the fly
        self.raiseAWarning('Trying to create a non-predefined plot of type ' + self.outStreamTypes[pltIndex] + '. If this fails, please refer to the and/or the related matplotlib method specification.')
        kwargs = {}
        for kk in plotSettings:
          if kk != 'attributes' and kk != self.outStreamTypes[pltIndex]:
            try:
              kwargs[kk] = ast.literal_eval(plotSettings[kk])
            except ValueError:
              kwargs[kk] = plotSettings[kk]
        try:
          if self.dim == 2:
            customFunctionCall = getattr(plt, self.outStreamTypes[pltIndex])
          else:
            customFunctionCall = getattr(self.plt3D, self.outStreamTypes[pltIndex])
          self.actPlot = customFunctionCall(**kwargs)
        except AttributeError as ae:
          self.raiseAnError(RuntimeError, '<' + str(ae) + '> -> in execution custom plot "' + self.outStreamTypes[pltIndex] + '" in Plot ' + self.name + '.\nSTREAM MANAGER: ERROR -> command has been called in the following way: ' + 'plt.' + self.outStreamTypes[pltIndex] + '(' + commandArgs + ')')

    if 'legend' in self.options['plotSettings']:
      plt.legend(**self.options['plotSettings']['legend'])

    # SHOW THE PICTURE
    self.__executeActions()
    plt.draw()
    # self.plt3D.draw(self.fig.canvas.renderer)

    if 'screen' in self.destinations and display:
      def handle_close(event):
        """
        This method is aimed to handle the closing of figures (overall when in interactive mode)
        @ In, event, instance, the event to close
        @ Out, None
        """
        self.fig.canvas.stop_event_loop()
        self.raiseAMessage('Closed Figure')
      self.fig.canvas.mpl_connect('close_event', handle_close)
      # self.plt.pause(1e-6)
      ## The following code is extracted from pyplot.pause without actually
      ## needing to force the code to sleep, according to MPL's documentation,
      ## this feature is experimental, hopefully by not calling the pause
      ## function, we can obtain consistent results.
      ## We are skipping a few of the sanity checks done in that function,
      ## since we are sure we have an interactive backend and access to the
      ## correct type of canvas and figure.
      self.fig.canvas.draw()
      plt.show(block=False)
      ## If your graphs are unresponsive to user input, you may want to consider
      ## adjusting this timeout, to allow more time for the input to be handled.
      self.fig.canvas.start_event_loop(1e-3)

      # self.fig.canvas.flush_events()

    for fileType in self.destinations:
      if fileType == 'screen':
        continue

      if not self.overwrite:
        prefix = str(self.counter) + '-'
      else:
        prefix = ''

      if len(self.filename) > 0:
        name = self.filename
      else:
        name = prefix + self.name + '_' + str(self.outStreamTypes).replace("'", "").replace("[", "").replace("]", "").replace(",", "-").replace(" ", "")

      if self.subDirectory is not None:
        name = os.path.join(self.subDirectory,name)

      plt.savefig(name + '.' + fileType, format = fileType)
    if 'screen' not in self.destinations:
      plt.close()
    gc.collect()
