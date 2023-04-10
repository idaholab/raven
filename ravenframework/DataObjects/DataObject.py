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
  Base class for both in-memory and in-disk data structures.
"""
import abc
import copy

from ..BaseClasses import BaseEntity
from ..utils import utils, InputData, InputTypes

class DataObjectsCollection(InputData.ParameterInput):
  """
    Class for reading in a collection of data objects.
  """
DataObjectsCollection.createClass("DataObjects")

class DataObject(utils.metaclass_insert(abc.ABCMeta, BaseEntity)):
  """
    Base class.  Data objects are RAVEN's method for storing data internally and passing it from one
    RAVEN entity to another.  Fundamentally, they consist of a collection of realizations, each of
    which contains inputs, outputs, and pointwise metadata.  In addition, the data object has global
    metadata.  The pointwise inputs and outputs could be floats, time-dependent, or ND-dependent variables.
    This base class is used to force the consistent API between all data containers
  """
  ### INPUT SPECIFICATION ###
  @classmethod
  def getInputSpecification(cls):
    """
      Get a reference to a class that specifies the input data for class "cls".
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying the input of cls.
    """
    inputSpecification = super(DataObject,cls).getInputSpecification()
    inputSpecification.addParam('hierarchical', InputTypes.BoolType)

    inputInput = InputData.parameterInputFactory('Input',contentType=InputTypes.StringType) #TODO list
    inputSpecification.addSub(inputInput)

    outputInput = InputData.parameterInputFactory('Output', contentType=InputTypes.StringType) #TODO list
    inputSpecification.addSub(outputInput)

    optionsInput = InputData.parameterInputFactory("options")
    for option in ['operator','pivotParameter']:
      optionSubInput = InputData.parameterInputFactory(option, contentType=InputTypes.StringType)
      optionsInput.addSub(optionSubInput)
    for option in ['inputRow','outputRow']:
      optionSubInput = InputData.parameterInputFactory(option, contentType=InputTypes.IntegerType)
      optionsInput.addSub(optionSubInput)
    for option in ['outputPivotValue','inputPivotValue']:
      optionSubInput = InputData.parameterInputFactory(option, contentType=InputTypes.FloatType)
      optionsInput.addSub(optionSubInput)
    inputSpecification.addSub(optionsInput)

    return inputSpecification

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.name             = 'DataObject'
    self.printTag         = self.name
    self._sampleTag       = 'RAVEN_sample_ID' # column name to track samples
    self.protectedTags    = ['RAVEN_parentID','RAVEN_isEnding'] # list(str) protected RAVEN variable names,
                                                                # should not be avail to user as var names
    self._inputs          = []     # list(str) if input variables
    self._inputsInitial   = []     # list(str) of initial input variables
    self._outputs         = []     # list(str) of output variables
    self._outputsInitial  = []     # list(str) of initial output variables
    self._metavars        = []     # list(str) of POINTWISE metadata variables
    self._orderedVars     = []     # list(str) of vars IN ORDER of their index
    self._orderedVarsInitial = []  # list(str) of initial vars IN ORDER of their index
    self._meta            = {}     # dictionary to collect meta until data is collapsed
    self._selectInput     = None   # if not None, describes how to collect input data from history
    self._selectOutput    = None   # if not None, describes how to collect output data from history
    self._pivotParams     = {}     # independent dimensions as keys, values are the vars that depend on them
    self._fromVarToIndex  = {}     # mapping between variables and indexes ({var:index}).
                                   #   "index" here refers to dimensional variables (e.g. time, x, y, z etc)
    self._aliases         = {}     # variable aliases
    self._data            = None   # underlying data structure
    self._collector       = None   # object used to collect samples
    self._inputKDTree     = None   # for finding outputs given inputs (pointset only?)
    self._scaleFactors    = None   # scaling factors inputs as {var:(mean,scale)}
    self.hierarchical     = False  # this flag controls the printing/plotting of the dataobject
                                   # in case it is an hierarchical one.
                                   # If True, all the branches are going to be printed/plotted
                                   # independently, otherwise they are going to be reconstructed
    self._tempPivotParam  = None   # temporary pivot parameter

  @property
  def sampleTag(self):
    """
      Getter property for _sampleTag, the tag that identifies the realization label for RAVEN
      @ In, None
      @ Out, sampleTag, string, variable name
    """
    return self._sampleTag

  def _readMoreXML(self, xmlNode):
    """
      Initializes data object based on XML input
      @ In, xmlNode, xml.etree.ElementTree.Element or InputData.ParameterInput specification, input information
      @ Out, None
    """
    if isinstance(xmlNode, InputData.ParameterInput):
      inp = xmlNode
    else:
      inp = DataObject.getInputSpecification()()
      inp.parseNode(xmlNode)

    # get hierarchical strategy
    self.hierarchical = inp.parameterValues.get("hierarchical", False)

    for child in inp.subparts:
      # TODO check for repeats, "notAllowdInputs", names in both input and output space
      if child.getName() == 'Input':
        self._inputs.extend(list(x.strip() for x in child.value.split(',') if x.strip()!=''))
      elif child.getName() == 'Output':
        self._outputs.extend(list(x.strip() for x in child.value.split(',') if x.strip()!=''))
      elif child.getName() == 'Index':
        depends = list(d.strip() for d in child.value.split(','))
        var = child.parameterValues['var']
        self._pivotParams[var] = depends
        if 'autogenerate' in child.parameterValues and child.parameterValues['autogenerate']:
          self._autogenerate.add(var)
      # options node
      elif child.getName() == 'options':
        duplicateInp = False # if True, then multiple specification options were used for input
        duplicateOut = False # if True, then multiple specification options were used for output
        for cchild in child.subparts:
          # pivot
          if cchild.getName() == 'pivotParameter':
            # TODO not applicable to ND, only to HistSet, but read it here
            # TODO add checks somewhere if both "index" and "pivotParameter" are provided
            self._tempPivotParam = cchild.value.strip()
          # input pickers
          elif cchild.getName() in ['inputRow','inputPivotValue']:
            if self._selectInput is not None:
              duplicateInp = True
            self.setSelectiveInput(cchild.getName(),cchild.value)
          # output pickers
          elif cchild.getName() in ['outputRow','outputPivotValue','operator']:
            if self._selectOutput is not None:
              duplicateOut = True
            self._selectOutput = (cchild.getName(),cchild.value)
        # TODO check this in the input checker instead of here?
        if duplicateInp:
          self.raiseAWarning('Multiple options were given to specify the input row to read! Using last entry:',self._selectInput)
        if duplicateOut:
          self.raiseAWarning('Multiple options were given to specify the output row to read! Using last entry:',self._selectOutput)
      # end options node
    # end input reading
    # clear keywords InputPlaceHolder but NOT the OutputPlaceHolder, for legacy reasons
    while 'InputPlaceHolder' in self._inputs:
      self._inputs.remove('InputPlaceHolder')
    # set default pivot parameters, if needed
    self._setDefaultPivotParams()
    # remove index variables from input/output spaces, but silently, since we'll still have them available later
    for index in self._pivotParams:
      try:
        self._outputs.remove(index)
      except ValueError:
        pass #not requested as output anyway
      try:
        self._inputs.remove(index)
      except ValueError:
        pass #not requested as input anyway
    # check inputs and outputs, if there were duplicates, error out
    dups = set(self._inputs).intersection(self._outputs)
    if dups:
      self.raiseAnError(IOError, 'Variables: "', ','.join(dups), '" are specified in both "Input" and "Output" Node of DataObject "', self.name,'"')
    self._orderedVars = self._inputs + self._outputs
    # check if protected vars have been violated
    if set(self.protectedTags).intersection(set(self._orderedVars)):
      self.raiseAnError(IOError, 'Input, Output and Index variables can not be part of RAVEN protected tags: '+','.join(self.protectedTags))
    self._inputsInitial = copy.copy(self._inputs)
    self._outputsInitial = copy.copy(self._outputs)
    self._orderedVarsInitial = copy.copy(self._orderedVars)

  def _setDefaultPivotParams(self):
    """
      Allows setting default pivot parameters.  In general, does nothing.
      @ In, None
      @ Out, None
    """
    pass

  def setPivotParams(self, params):
    """
      Sets the pivot parameters for variables.
      @ In, params, dict, var:[params] as str:list(str)
      @ Out, None
    """
    # TODO typechecking, assertions
    coords = set().union(*params.values())
    for coord in coords:
      if coord not in self._pivotParams:
        self._pivotParams[coord] = list(var for var in params.keys() if coord in params[var])
      else:
        self._pivotParams[coord] = list(set(list(var for var in params.keys() if
                                                 coord in params[var]) + self._pivotParams[coord]))

  def setSelectiveInput(self, option, value):
    """
      Sets the input selection method for retreiving subset data.
      @ In, option, str, from [inputRow,inputPivotValue]
      @ In, value, int or float, either the index (row number) or the pivot value (will be cast if other type)
      @ Out, None
    """
    assert(option in ['inputRow','inputPivotValue'])
    if option == 'inputRow':
      value = int(value)
    elif option == 'inputPivotValue':
      value = float(value)
    self._selectInput = (option,value)
    self.raiseADebug('Set selective input to',self._selectInput)

  def setSelectiveOutput(self, option, value):
    """
      Sets the output selection method for retrieving subset data.
      @ In, option, str, from [outputRow,outputPivotValue,operator]
      @ In, value, int or float or str, index or pivot value or operator name respectively
      @ Out, None
    """
    assert(option in ['outputRow','outputPivotValue','operator'])
    if option == 'outputRow':
      value = int(value)
    elif option == 'outputPivotValue':
      value = float(value)
    elif option == 'operator':
      value = value.strip().lower()
    self._selectOutput = (option,value)
    self.raiseADebug('Set selective output to',self._selectOutput)

  def flush(self):
    """
      Resets the DataObject used for output to its initial condition in order to run a RAVEN
      workflow after one has already been run.
      @ In, None
      @ Out, None
    """
    self._data = None
    self._metavars = []
    self._orderedVars = copy.copy(self._orderedVarsInitial)
    self._inputs = copy.copy(self._inputsInitial)
    self._outputs = copy.copy(self._outputsInitial)
    self._meta = {}
    self._collector = None
    self._scaleFactors = {}

  ######################
  # DATA CONTAINER API #
  ######################
  @abc.abstractmethod
  def addExpectedMeta(self,keys, params={}):
    """
      Registers meta to look for in realization
      @ In, keys, set(str), keys to register
      @ In, params, dict, optional, {key:[indexes]}, keys of the dictionary are the variable names,
        values of the dictionary are lists of the corresponding indexes/coordinates of given variable
      @ Out, None
    """
    pass

  @abc.abstractmethod
  def addMeta(self, tag, xmlDict):
    """
      Adds general (not pointwise) metadata to this data object.  Can add several values at once, collected
      as a dict keyed by target variables.
      Data ends up being written as follows (see docstrings above for dict structure)
       - A good default for 'target' is 'general' if there's not a specific target
      <tag>
        <target>
          <scalarMetric>value</scalarMetric>
          <scalarMetric>value</scalarMetric>
          <vectorMetric>
            <wrt>value</wrt>
            <wrt>value</wrt>
          </vectorMetric>
        </target>
        <target>
          <scalarMetric>value</scalarMetric>
          <vectorMetric>
            <wrt>value</wrt>
          </vectorMetric>
        </target>
      </tag>
      @ In, tag, str, section to add metadata to, usually the data submitter (BasicStatistics, DataObject, etc)
      @ In, xmlDict, dict, data to change, of the form {target:{scalarMetric:value,scalarMetric:value,vectorMetric:{wrt:value,wrt:value}}}
      @ Out, None
    """
    pass

  @abc.abstractmethod
  def addRealization(self, rlz):
    """
      Adds a "row" (or "sample") to this data object.
      This is the method to add data to this data object.
      Note that rlz can include many more variables than this data object actually wants.
      Before actually adding the realization, data is formatted for this data object.
      @ In, rlz, dict, {var:val} format where
                         "var" is the variable name as a string,
                         "val" is either a float or a np.ndarray of values.
      @ Out, None
    """
    pass

  @abc.abstractmethod
  def addVariable(self, varName, values, classify='meta'):
    """
      Adds a variable/column to the data.  "values" needs to be as long as self.size.
      @ In, varName, str, name of new variable
      @ In, values, np.array, new values (floats/str for scalars, xr.DataArray for hists)
      @ In, classify, str, optional, either 'input', 'output', or 'meta'
      @ Out, None
    """
    pass

  @abc.abstractmethod
  def asDataset(self):
    """
      Casts this dataobject as an xr.Dataset.
      Functionally, typically collects the data from self._collector and places it in self._data.
      Efficiency note: this is the slowest part of typical data collection.
      @ In, None
      @ Out, xarray.Dataset, all the data from this data object.
    """
    pass

  @abc.abstractmethod
  def constructNDSample(self, vals, dims, coords, name=None):
    """
      Constructs a single realization instance (for one variable) from a realization entry.
      @ In, vals, np.ndarray, should have shape of (len(coords[d]) for d in dims)
      @ In, dims, list(str), names of dependent dimensions IN ORDER of appearance in vals, e.g. ['time','x','y']
      @ In, coords, dict, {dimension:list(float)}, values for each dimension at which 'val' was obtained, e.g. {'time':
      @ Out, obj, xr.DataArray, completed realization instance suitable for sending to "addRealization"
    """
    pass

  @abc.abstractmethod
  def getDimensions(self, var):
    """
      Provides the independent dimensions that this variable depends on.
      To get all dimensions at once, use self.indexes property.
      @ In, var, str, name of variable (if None, give all)
      @ Out, dims, dict, {name:values} of independent dimensions
    """
    pass

  @abc.abstractmethod
  def getMeta(self, keys=None, pointwise=False, general=False):
    """
      Method to obtain entries in the metadata.  If niether pointwise nor general, then returns an empty dict.
       @ In, keys, list(str), optional, the keys (or main tag) to search for.  If None, return all.
       @ In, pointwise, bool, optional, if True then matches will be searched in the pointwise metadata
       @ In, general, bool, optional, if True then matches will be searched in the general metadata
       @ Out, meta, dict, key variables/xpaths to data object entries (column if pointwise, XML if general)
    """
    pass

  @abc.abstractmethod
  def getVars(self, subset=None):
    """
      Gives list of variables that are part of this dataset.
      @ In, subset, str, optional, if given can return 'input','output','meta' subset types
      @ Out, getVars, list(str), list of variable names requested
    """
    pass

  @abc.abstractmethod
  def getVarValues(self, var):
    """
      Returns the sampled values of "var"
      @ In, var, str or list(str), name(s) of variable(s)
      @ Out, res, xr.DataArray, samples (or dict of {var:xr.DataArray} if multiple variables requested)
    """
    pass

  @abc.abstractmethod
  def realization(self, index=None, matchDict=None, noMatchDict=None, tol=1e-15, unpackXArray=False, asDataSet = False, options = None):
    """
      Method to obtain a realization from the data, either by index or matching value.
      Either "index" or one of ("matchDict", "noMatchDict") must be supplied.
      If matchDict and no match is found, will return (len(self),None) after the pattern of numpy, scipy
      @ In, index, int, optional, number of row to retrieve (by index, not be "sample")
      @ In, matchDict, dict, optional, {key:val} to search for matches
      @ In, noMatchDict, dict, optional, {key:val} to search for antimatches (vars should NOT match vals within tolerance)
      @ In, asDataSet, bool, optional, return realization from the data as a DataSet
      @ In, tol, float, optional, tolerance to which match should be made
      @ In, unpackXArray, bool, optional, True if the coordinates of the xarray variables must be exposed in the dict (e.g. if P(t) => {P:ndarray, t:ndarray}) (valid only for dataset)
      @ In, options, dict, optional, options to be applied to the search
      @ Out, index, int, optional, index where found (or len(self) if not found), only returned if matchDict
      @ Out, rlz, dict, realization requested (None if not found)
    """
    pass

  @abc.abstractmethod
  def load(self, fname, style='netCDF', **kwargs):
    """
      Reads this dataset from disk based on the format.
      @ In, fname, str, path and name of file to read
      @ In, style, str, optional, options are enumerated below
      @ In, kwargs, dict, optional, additional arguments to pass to reading function
      @ Out, None
    """
    pass

  @abc.abstractmethod
  def remove(self, realization=None, variable=None):
    """
      Used to remove either a realization or a variable from this data object.
      @ In, realization, dict or int, optional, (matching or index of) realization to remove
      @ In, variable, str, optional, name of "column" to remove
      @ Out, None
    """
    pass

  @abc.abstractmethod
  def reset(self):
    """
      Sets this object back to its initial state.
      @ In, None
      @ Out, None
    """
    pass

  @abc.abstractmethod
  def sliceByIndex(self, index):
    """
      Returns list of realizations at "snapshots" along "axis"
      @ In, index, str, name of index along which to obtain slices
      @ Out, slices, list, list of slices
    """
    pass

  @abc.abstractmethod
  def write(self, fname, style='netCDF', **kwargs):
    """
      Writes this dataset to disk based on the format.
      @ In, fname, str, path and name of file to write
      @ In, style, str, optional, options are enumerated below
      @ In, kwargs, dict, optional, additional arguments to pass to writing function
      @ Out, None
    """
    pass
