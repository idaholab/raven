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
Created on July 10, 2013

@author: alfoa
"""
#External Modules------------------------------------------------------------------------------------
import numpy as np
import time
import sys
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .PostProcessorInterface import PostProcessorInterface
from ...utils import InputData, InputTypes
from ... import Files
#Internal Modules End-----------------------------------------------------------


class TopologicalDecomposition(PostProcessorInterface):
  """
    TopologicalDecomposition class - Computes an approximated hierarchical
    Morse-Smale decomposition from an input point cloud consisting of an
    arbitrary number of input parameters and a response value per input point
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    ## This will replace the lines above
    inputSpecification = super().getInputSpecification()

    TDGraphInput = InputData.parameterInputFactory("graph", contentType=InputTypes.StringType)
    inputSpecification.addSub(TDGraphInput)

    TDGradientInput = InputData.parameterInputFactory("gradient", contentType=InputTypes.StringType)
    inputSpecification.addSub(TDGradientInput)

    TDBetaInput = InputData.parameterInputFactory("beta", contentType=InputTypes.FloatType)
    inputSpecification.addSub(TDBetaInput)

    TDKNNInput = InputData.parameterInputFactory("knn", contentType=InputTypes.IntegerType)
    inputSpecification.addSub(TDKNNInput)

    TDWeightedInput = InputData.parameterInputFactory("weighted", contentType=InputTypes.StringType) #bool
    inputSpecification.addSub(TDWeightedInput)

    TDInteractiveInput = InputData.parameterInputFactory("interactive", contentType=InputTypes.StringType) #bool
    inputSpecification.addSub(TDInteractiveInput)

    TDPersistenceInput = InputData.parameterInputFactory("persistence", contentType=InputTypes.StringType)
    inputSpecification.addSub(TDPersistenceInput)

    TDSimplificationInput = InputData.parameterInputFactory("simplification", contentType=InputTypes.FloatType)
    inputSpecification.addSub(TDSimplificationInput)

    TDParametersInput = InputData.parameterInputFactory("parameters", contentType=InputTypes.StringType)
    inputSpecification.addSub(TDParametersInput)

    TDResponseInput = InputData.parameterInputFactory("response", contentType=InputTypes.StringType)
    inputSpecification.addSub(TDResponseInput)

    TDNormalizationInput = InputData.parameterInputFactory("normalization", contentType=InputTypes.StringType)
    inputSpecification.addSub(TDNormalizationInput)

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.acceptedGraphParam = ['approximate knn', 'delaunay', 'beta skeleton', \
                               'relaxed beta skeleton']
    self.acceptedPersistenceParam = ['difference','probability','count']#,'area']
    self.acceptedGradientParam = ['steepest', 'maxflow']
    self.acceptedNormalizationParam = ['feature', 'zscore', 'none']

    # Some default arguments
    self.gradient = 'steepest'
    self.graph = 'beta skeleton'
    self.beta = 1
    self.knn = -1
    self.simplification = 0
    self.persistence = 'difference'
    self.normalization = None
    self.weighted = False
    self.parameters = {}

  def inputToInternal(self, currentInp):
    """
      Function to convert the incoming input into a usable format
      @ In, currentInp, list or DataObjects, The input object to process
      @ Out, inputDict, dict, the converted input
    """
    # TODO typechecking against what currentInp can be; so far it's a length=1 list with a dataobject inside
    currentInp = currentInp[0]
    currentInp.asDataset()
    # nowadays, our only input should be DataObject
    ## if no "type", then you're not a PointSet or HistorySet
    if not hasattr(currentInp,'type') or currentInp.type != 'PointSet':
      self.raiseAnError(IOError, self.__class__.__name__,
                        ' postprocessor only accepts PointSet DataObjects for input. ',
                        ' Requested: ', type(currentInp))
    # now we know we have a PointSet
    ## TODO FIXME maintaining old structure for now, in the future convert to use DataObject directly
    ##    and not bother with inputToInternal
    ##    This works particularly well since we only accept point sets.
    data = currentInp.asDataset(outType='dict')['data']
    inputDict = {'features':dict((var,data[var]) for var in self.parameters['features']),
                 'targets' :dict((var,data[var]) for var in self.parameters['targets' ]),
                 'metadata':currentInp.getMeta(general=True)}
    #if 'PointProbability' in currentInp.getVars():
    inputDict['metadata']['PointProbability'] = currentInp.getVarValues('PointProbability').values
    #else:
    #  raise NotImplementedError # TODO
    return inputDict

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    for child in paramInput.subparts:
      if child.getName() == "graph":
        self.graph = child.value.lower()
        if self.graph not in self.acceptedGraphParam:
          self.raiseAnError(IOError, 'Requested unknown graph type: ',
                            self.graph, '. Available options: ',
                            self.acceptedGraphParam)
      elif child.getName() == "gradient":
        self.gradient = child.value.lower()
        if self.gradient not in self.acceptedGradientParam:
          self.raiseAnError(IOError, 'Requested unknown gradient method: ',
                            self.gradient, '. Available options: ',
                            self.acceptedGradientParam)
      elif child.getName() == "beta":
        self.beta = child.value
        if self.beta <= 0 or self.beta > 2:
          self.raiseAnError(IOError, 'Requested invalid beta value: ',
                            self.beta, '. Allowable range: (0,2]')
      elif child.getName() == 'knn':
        self.knn = child.value
      elif child.getName() == 'simplification':
        self.simplification = child.value
      elif child.getName() == 'persistence':
        self.persistence = child.value.lower()
        if self.persistence not in self.acceptedPersistenceParam:
          self.raiseAnError(IOError, 'Requested unknown persistence method: ',
                            self.persistence, '. Available options: ',
                            self.acceptedPersistenceParam)
      elif child.getName() == 'parameters':
        self.parameters['features'] = child.value.strip().split(',')
        for i, parameter in enumerate(self.parameters['features']):
          self.parameters['features'][i] = self.parameters['features'][i]
      elif child.getName() == 'weighted':
        self.weighted = child.value in ['True', 'true']
      elif child.getName() == 'response':
        self.parameters['targets'] = child.value
      elif child.getName() == 'normalization':
        self.normalization = child.value.lower()
        if self.normalization not in self.acceptedNormalizationParam:
          self.raiseAnError(IOError, 'Requested unknown normalization type: ',
                            self.normalization, '. Available options: ',
                            self.acceptedNormalizationParam)
    # register metadata
    self.addMetaKeys(['maxLabel','minLabel'])

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    inputList,outputDict = evaluation

    if output.type == 'PointSet':
      # TODO this is a slow dict-based implementation.  It should be improved on need.
      # TODO can inputList ever be multiple dataobjects?
      if len(inputList) > 1:
        self.raiseAnError(NotImplementedError, 'Need to implement looping over all inputs.')
      fromInput = inputList[0].asDataset('dict')['data']
      results = dict((var,fromInput[var]) for var in output.getVars() if var in fromInput.keys())
      for label in ['minLabel','maxLabel']:
        results[label] = outputDict[label]
      output.load(results,style='dict')
      output.addMeta(self.type,{'general':{'hierarchy':outputDict['hierarchy']}})
      return


      #### OLD ####
      requestedInput = output.getParaKeys('input')
      requestedOutput = output.getParaKeys('output')
      dataLength = None
      for inputData in inputList:
        # Pass inputs from input data to output data
        for key, value in inputData.getParametersValues('input').items():
          if key in requestedInput:
            # We need the size to ensure the data size is consistent, but there
            # is no guarantee the data is not scalar, so this check is necessary
            myLength = 1
            if hasattr(value, "__len__"):
              myLength = len(value)

            if dataLength is None:
              dataLength = myLength
            elif dataLength != myLength:
              dataLength = max(dataLength, myLength)
              self.raiseAWarning('Data size is inconsistent. Currently set to '
                                 + str(dataLength) + '.')

            for val in value:
              output.updateInputValue(key, val)

        # Pass outputs from input data to output data
        for key, value in inputData.getParametersValues('output').items():
          if key in requestedOutput:
            # We need the size to ensure the data size is consistent, but there
            # is no guarantee the data is not scalar, so this check is necessary
            myLength = 1
            if hasattr(value, "__len__"):
              myLength = len(value)

            if dataLength is None:
              dataLength = myLength
            elif dataLength != myLength:
              dataLength = max(dataLength, myLength)
              self.raiseAWarning('Data size is inconsistent. Currently set to '
                                      + str(dataLength) + '.')

            for val in value:
              output.updateOutputValue(key, val)

        # Append the min/max labels to the data whether the user wants them or
        # not, and place the hierarchy information into the metadata
        for key, values in outputDict.items():
          if key in ['minLabel', 'maxLabel']:
            for value in values:
              output.updateOutputValue(key, [value])
          elif key in ['hierarchy']:
            output.updateMetadata(key, [values])
    else:
      self.raiseAWarning('Output type ' + type(output).__name__ + ' not'
                         + ' yet implemented. I am going to skip it.')

  def userInteraction(self):
    """
      A placeholder for allowing user's to interact and tweak the model in-situ
      before saving the analysis results
      @ In, None
      @ Out, None
    """
    pass

  def run(self, inputIn):
    """
      Function to finalize the filter => execute the filtering
      @ In, inputIn, dict, dictionary of data to process
      @ Out, outputDict, dict, Dictionary containing the post-processed results
    """

    internalInput = self.inputToInternal(inputIn)
    outputDict = {}

    myDataIn = internalInput['features']
    myDataOut = internalInput['targets']

    self.outputData = myDataOut[self.parameters['targets']]
    self.pointCount = len(self.outputData)
    self.dimensionCount = len(self.parameters['features'])

    self.inputData = np.zeros((self.pointCount, self.dimensionCount))
    for i, lbl in enumerate(self.parameters['features']):
      self.inputData[:, i] = myDataIn[lbl]

    if self.weighted:
      self.weights = internalInput['metadata']['PointProbability']
    else:
      self.weights = None

    self.names = self.parameters['features'] + [self.parameters['targets']]

    self.__amsc = None
    self.userInteraction()

    ## Possibly load this here in case people have trouble building it, so it
    ## only errors if they try to use it?
    from AMSC.AMSC_Object import AMSC_Object

    if self.__amsc is None:
      self.__amsc = AMSC_Object(X=self.inputData, Y=self.outputData,
                                w=self.weights, names=self.names,
                                graph=self.graph, gradient=self.gradient,
                                knn=self.knn, beta=self.beta,
                                normalization=self.normalization,
                                persistence=self.persistence, debug=False)

    self.__amsc.Persistence(self.simplification)
    partitions = self.__amsc.Partitions()

    outputDict['minLabel'] = np.zeros(self.pointCount)
    outputDict['maxLabel'] = np.zeros(self.pointCount)
    for extPair, indices in partitions.items():
      for idx in indices:
        outputDict['minLabel'][idx] = extPair[0]
        outputDict['maxLabel'][idx] = extPair[1]
    outputDict['hierarchy'] = self.__amsc.PrintHierarchy()
    self.__amsc.BuildModels()
    linearFits = self.__amsc.SegmentFitCoefficients()
    linearFitnesses = self.__amsc.SegmentFitnesses()

    for key in linearFits.keys():
      coefficients = linearFits[key]
      rSquared = linearFitnesses[key]
      outputDict['coefficients_%d_%d' % (key[0], key[1])] = coefficients
      outputDict['R2_%d_%d' % (key[0], key[1])] = rSquared

    return outputDict

try:
  import PySide.QtCore as qtc
  __QtAvailable = True
except ImportError as e:
  try:
    import PySide2.QtCore as qtc
    __QtAvailable = True
  except ImportError as e:
    __QtAvailable = False

if __QtAvailable:
  class mQTopologicalDecomposition(type(TopologicalDecomposition), type(qtc.QObject)):
    """
      Class used to solve the metaclass conflict
    """
    pass

  class QTopologicalDecomposition(TopologicalDecomposition, qtc.QObject, metaclass=mQTopologicalDecomposition):
    """
      TopologicalDecomposition class - Computes an approximated hierarchical
      Morse-Smale decomposition from an input point cloud consisting of an
      arbitrary number of input parameters and a response value per input point
    """
    requestUI = qtc.Signal(str,str,dict)
    @classmethod
    def getInputSpecification(cls):
      """
        Method to get a reference to a class that specifies the input data for
        class cls.
        @ In, cls, the class for which we are retrieving the specification
        @ Out, inputSpecification, InputData.ParameterInput, class to use for
          specifying input of cls.
      """
      inputSpecification = super(QTopologicalDecomposition, cls).getInputSpecification()
      inputSpecification.addSub(InputData.parameterInputFactory("interactive"))
      return inputSpecification

    def __init__(self):
      """
       Constructor
       @ In, None
       @ Out, None
      """
      super().__init__()
      # TopologicalDecomposition.__init__(self)
      # qtc.QObject.__init__(self)

      self.interactive = False
      self.uiDone = True ## If it has not been requested, then we are not waiting for a UI

    def _localWhatDoINeed(self):
      """
        This method is a local mirror of the general whatDoINeed method.
        It is implemented by the samplers that need to request special objects
        @ In , None, None
        @ Out, needDict, list of objects needed
      """
      return {'internal':[(None,'app')]}

    def _localGenerateAssembler(self,initDict):
      """
        Generates the assembler.
        @ In, initDict, dict of init objects
        @ Out, None
      """
      self.app = initDict['internal']['app']
      if self.app is None:
        self.interactive = False

    def _localReadMoreXML(self, xmlNode):
      """
        Function to grab the names of the methods this post-processor will be
        using
        @ In, xmlNode    : Xml element node
        @ Out, None
      """
      paramInput = QTopologicalDecomposition.getInputSpecification()()
      paramInput.parseNode(xmlNode)
      self._handleInput(paramInput)

    def _handleInput(self, paramInput):
      """
        Function to handle the parsed paramInput for this class.
        @ In, paramInput, ParameterInput, the already parsed input.
        @ Out, None
      """
      TopologicalDecomposition._handleInput(self, paramInput)
      for child in paramInput.subparts:
        if child.getName() == 'interactive':
          self.interactive = True

    def userInteraction(self):
      """
        Launches an interface allowing the user to tweak specific model
        parameters before saving the results to the output object(s).
        @ In, None
        @ Out, None
      """
      self.uiDone = not self.interactive

      if self.interactive:
        ## Connect our own signal to the slot on the main thread
        self.requestUI.connect(self.app.createUI)

        ## Connect our own slot to listen for whenver the main thread signals a
        ## window has been closed
        self.app.windowClosed.connect(self.signalDone)

        ## Give this UI a unique id in case other threads are requesting UI
        ##  elements
        uiID = str(id(self))

        ## Send the request for a UI thread to the main application
        self.requestUI.emit('TopologyWindow', uiID,
                            {'X':self.inputData, 'Y':self.outputData,
                             'w':self.weights, 'names':self.names,
                             'graph':self.graph, 'gradient': self.gradient,
                             'knn':self.knn, 'beta':self.beta,
                             'normalization':self.normalization,
                             'views': ['TopologyMapView', 'SensitivityView',
                                       'FitnessView', 'ScatterView2D',
                                       'ScatterView3D']})

        ## Spinlock will wait until this instance's window has been closed
        while(not self.uiDone):
          time.sleep(1)

        ## First check that the requested UI exists, and then if that UI has the
        ## requested information, if not proceed as if it were not an
        ## interactive session.
        if uiID in self.app.UIs and hasattr(self.app.UIs[uiID],'amsc'):
          self.__amsc = self.app.UIs[uiID].amsc
          self.simplification = self.app.UIs[uiID].amsc.Persistence()
        else:
          self.__amsc = None

    def signalDone(self,uiID):
      """
        In Qt language, this is a slot that will accept a signal from the UI
        saying that it has completed, thus allowing the computation to begin
        again with information updated by the user in the UI.
        @In, uiID, string, the ID of the user interface that signaled its
            completion. Thus, if several UI windows are open, we don't proceed,
            until the correct one has signaled it is done.
        @Out, None
      """
      if uiID == str(id(self)):
        self.uiDone = True
