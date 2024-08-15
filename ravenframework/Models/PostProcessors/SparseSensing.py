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
'''
  Created on May 24, 2022
  @ Authors: Mohammad Abdo (@Jimmy-INL)
             Niharika Karnik (@nkarnik)
'''
import pysensors as ps
import pandas as pd
import numpy as np
import xarray as xr

from .PostProcessorReadyInterface import PostProcessorReadyInterface
from ...utils import InputData, InputTypes

class SparseSensing(PostProcessorReadyInterface):
  """
    This Postprocessor class finds the optimal locations of sparse sensors for both classification and reconstruction problems in the presence of constrained locations for sensing.
    The implemention utilizes the opensource library pysensors and is based on following publications:
    - Brunton, Bingni W., et al. "Sparse sensor placement optimization for classification." SIAM Journal on Applied Mathematics 76.5 (2016): 2099-2122.
    - Manohar, Krithika, et al. "Data-driven sparse sensor placement for reconstruction: Demonstrating the benefits of exploiting known patterns." IEEE Control Systems Magazine 38.3 (2018): 63-86.
    - de Silva, Brian M., et al. "PySensors: A Python package for sparse sensor placement." arXiv preprint arXiv:2102.13476 (2021).
    - Karnik, Niharika, et al. " Constrained optimization of sensor placement for nuclear digital twins." IEEE Sensors Journal (2024).
  """
  goalsDict = {'reconstruction':r"""Sparse sensor placement Optimization for Reconstruction (SSPOR)""",
          'classification':r"""Sparse sensor placement Optimization for Classification (SSPOC)"""}
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(SparseSensing, cls).getInputSpecification()
    goal = InputData.parameterInputFactory('Goal',
                                                  printPriority=108,
                                                  descr=r"""The goal of the sparse sensor optimization (i.e., reconstruction or classification)""")
    goal.addParam("subType", InputTypes.makeEnumType("Goal", "GoalType", ['reconstruction','classification']), False, default='reconstruction')
    inputSpecification.addSub(goal)
    features = InputData.parameterInputFactory("features", contentType=InputTypes.StringListType,
                                                printPriority=108,
                                                descr=r"""Features/inputs of the data model""")
    goal.addSub(features)
    xValue = InputData.parameterInputFactory("xValue", contentType=InputTypes.StringListType,
                                                printPriority=108,
                                                descr=r"""Variable plotted on the X-axis of the data model""")
    goal.addSub(xValue)
    yValue = InputData.parameterInputFactory("yValue", contentType=InputTypes.StringListType,
                                                printPriority=108,
                                                descr=r"""Variable plotted on the Y-axis of the data model""")
    goal.addSub(yValue)
    zValue = InputData.parameterInputFactory("yValue", contentType=InputTypes.StringListType,
                                                printPriority=108,
                                                descr=r"""Variable plotted on the Z-axis of the data model""")
    goal.addSub(zValue)
    measuredState = InputData.parameterInputFactory("measuredState", contentType=InputTypes.StringType,
                                                printPriority=108,
                                                descr=r"""State Variable to be measured/sensed""")
    goal.addSub(measuredState)
    labels = InputData.parameterInputFactory("labels", contentType=InputTypes.StringType,
                                                printPriority=108,
                                                descr=r"""labels/target for the classification case""")
    goal.addSub(labels)
    basis = InputData.parameterInputFactory("basis", contentType=InputTypes.makeEnumType("basis","basis Type",['Identity','SVD','RandomProjetion']),
                                                           printPriority=108,
                                                           descr=r"""The type of basis onto which the data are projected""", default='SVD')
    goal.addSub(basis)
    nModes = InputData.parameterInputFactory("nModes", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""The number of modes retained""")
    goal.addSub(nModes)
    nSensors = InputData.parameterInputFactory("nSensors", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""The number of sensors used""")
    goal.addSub(nSensors)
    nConstSensors = InputData.parameterInputFactory("nConstSensors", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""The number of constraint sensors for GQR optimizer""")
    goal.addSub(nConstSensors)
    constraintOption = InputData.parameterInputFactory("constraintOption", contentType=InputTypes.makeEnumType("Constraint","ConstraintOption",["max_n","exact_n","predetermined"]),
                                                printPriority=108,
                                                descr=r"""The constraint the user wants to implement (max_n, exact_n, predetermined)""")
    goal.addSub(constraintOption)
    optimizer = InputData.parameterInputFactory("optimizer", contentType=InputTypes.makeEnumType("optimizer","optimizer type",['QR', 'GQR']),
                                                           printPriority=108,
                                                           descr=r"""The type of optimizer used""",default='QR')
    goal.addSub(optimizer)
    classifier = InputData.parameterInputFactory("classifier", contentType=InputTypes.makeEnumType("classifier","classifier type",['LDA']),
                                                           printPriority=108,
                                                           descr=r"""The type of classifier used""",default='LDA')
    goal.addSub(classifier)
    seed = InputData.parameterInputFactory("seed", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""The integer seed use for sensor placement random number seed""")
    goal.addSub(seed)
    ConstrainedRegions = InputData.parameterInputFactory("ConstrainedRegions", contentType=InputTypes.StringType,
                                                printPriority=108,
                                                descr=r"""The shape of region we want to constrain""")
    goal.addSub(ConstrainedRegions)
    ConstrainedRegions.addParam("type", InputTypes.makeEnumType("Constraint","ConstraintType",['Circle','Ellipse','Line','Polygon','Parabola']), True,
                       descr="type of Constrained Region shape the user wants- (Constraint can be a circle, parabola, ellipse, line, rectangle, square or user defined constraint too)")
    centerX = InputData.parameterInputFactory("centerX", contentType=InputTypes.FloatType,
                                                           printPriority=108,
                                                           descr=r"""The x co-ordinate of center of circle,ellipse""")
    ConstrainedRegions.addSub(centerX)
    centerY = InputData.parameterInputFactory("centerY", contentType=InputTypes.FloatType,
                                                           printPriority=108,
                                                           descr=r"""The y co-ordinate of center of circle,ellipse""")
    ConstrainedRegions.addSub(centerY)
    radius = InputData.parameterInputFactory("radius", contentType=InputTypes.FloatType,
                                                           printPriority=108,
                                                           descr=r"""The radius of circle,ellipse""")
    ConstrainedRegions.addSub(radius)
    loc = InputData.parameterInputFactory("loc", contentType=InputTypes.makeEnumType("Constraint","Constraint Locations",['in','out']),
                                                           printPriority=108,
                                                           descr=r"""Whether the constraint region is inside or outside the shape defined as constraint""")
    ConstrainedRegions.addSub(loc)
    width = InputData.parameterInputFactory("width", contentType=InputTypes.FloatType,
                                                           printPriority=108,
                                                           descr=r"""Width of the ellipse""")
    ConstrainedRegions.addSub(width)
    height = InputData.parameterInputFactory("height", contentType=InputTypes.FloatType,
                                                           printPriority=108,
                                                           descr=r"""Height of the ellipse""")
    ConstrainedRegions.addSub(height)
    angle = InputData.parameterInputFactory("angle", contentType=InputTypes.FloatType,
                                                           printPriority=108,
                                                           descr=r"""Angle of rotation of the ellipse""")
    ConstrainedRegions.addSub(angle)
    x1 = InputData.parameterInputFactory("x1", contentType=InputTypes.FloatType,
                                                           printPriority=108,
                                                           descr=r"""X co-ordinate of one of the points that defines the line""")
    ConstrainedRegions.addSub(x1)
    x2 = InputData.parameterInputFactory("x2", contentType=InputTypes.FloatType,
                                                           printPriority=108,
                                                           descr=r"""X co-ordinate of the other point that defines the line""")
    ConstrainedRegions.addSub(x2)
    y1 = InputData.parameterInputFactory("y1", contentType=InputTypes.FloatType,
                                                           printPriority=108,
                                                           descr=r"""Y co-ordinate of one of the points that defines the line""")
    ConstrainedRegions.addSub(y1)
    y2 = InputData.parameterInputFactory("y2", contentType=InputTypes.FloatType,
                                                           printPriority=108,
                                                           descr=r"""Y co-ordinate of the other point that defines the line""")
    ConstrainedRegions.addSub(y2)
    h = InputData.parameterInputFactory("h", contentType=InputTypes.FloatType,
                                                           printPriority=108,
                                                           descr=r"""X coordinate of the vertex of the parabola we want to be constrained""")
    ConstrainedRegions.addSub(h)
    k = InputData.parameterInputFactory("k", contentType=InputTypes.FloatType,
                                                           printPriority=108,
                                                           descr=r""" Y coordinate of the vertex of the parabola we want to be constrained""")
    ConstrainedRegions.addSub(k)
    a = InputData.parameterInputFactory("a", contentType=InputTypes.FloatType,
                                                           printPriority=108,
                                                           descr=r"""X coordinate of the focus of the parabola""")
    ConstrainedRegions.addSub(a)
    xyCoords = InputData.parameterInputFactory("xyCoords", contentType=InputTypes.FloatListType,
                                                           printPriority=108,
                                                           descr=r"""an array consisting of tuples for (x,y) coordinates of points of the Polygon where N = No. of sides of the polygon""")
    ConstrainedRegions.addSub(xyCoords)
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.setInputDataType('xrDataset')
    self.keepInputMeta(False)
    self.outputMultipleRealizations = True                   # True indicate multiple realizations are returned
    self.pivotParameter = None                               # time-dependent data pivot parameter. None if the problem is steady state
    self.validDataType = ['PointSet','HistorySet','DataSet'] # FIXME: Should remove the unsupported ones
    self.sparseSensingGoal = None                            # The goal of the sensor selection. i.e., reconstruction or classification
    self.nSensors = None                                     # The number of the sensors required by the user.
    self.nConstSensors = None                                # The number of sensors in the constrained region
    self.constraintOption = None                             # The constraint name that the user wants to implement.
    self.ConstrainedRegions = None                           # The shape of region we want to constrain
    self.nModes = None                                       # The number of modes/basis used to truncate the singular value decomposition
    self.basis = None                                        # The types of basis used in the projection. i.e., SVD, Identity, or Random Projection
    self.sensingFeatures = None                              # The variable representing the features of the data i.e., X, Y, SensorID, etc.
    self.sensingStateVariable = None                         # The variable representing the state
    self.sensingLabels = None                                # The Response of interest to be reconstructed (or classify)
    self.optimizer = None                                    # The Optimizer type used in the Sparse sensing selection (default: QR)
    self.classifier = None                                   # The classifier type used in the Sparse sensing selection (default: LinearDiscriminantAnalysis)
    self.sampleTag = 'RAVEN_sample_ID'                       # The sample tag
    self.centerX = None                                     # The x co-ordinate of center of circle,ellipse
    self.centerY = None                                     # The y co-ordinate of center of circle,ellipse
    self.radius = None                                       # The radius of circle,ellipse
    self.loc = None                                          # Whether the constraint region is inside or outside the shape defined as constraint
    self.width = None                                        # Width of the ellipse
    self.height = None                                       # Height of the ellipse
    self.angle = None                                        # Angle of rotation of the ellipse
    self.x1 = None                                           # X co-ordinate of one of the points that defines the line
    self.x2 = None                                           # X co-ordinate of the other point that defines the line
    self.y1 = None                                           # Y co-ordinate of one of the points that defines the line
    self.y2 = None                                           # Y co-ordinate of the other point that defines the line
    self.h = None                                            # X coordinate of the vertex of the parabola we want to be constrained
    self.k = None                                            # Y coordinate of the vertex of the parabola we want to be constrained
    self.a = None                                            # X coordinate of the focus of the parabola
    self.xyCoords = None                                    # an array consisting of tuples for (x,y) coordinates of points of the Polygon where N = No. of sides of the polygon

  def initialize(self, runInfo, inputs, initDict=None):
    """
      Method to initialize the DataClassifier post-processor.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, optional, dictionary with initialization options
      @ Out, None
    """
    super().initialize(runInfo, inputs, initDict)
    if len(inputs)>1:
      self.raiseAnError(IOError, 'Post-Processor', self.name, 'accepts only one dataObject')

  def _handleInput(self, paramInput):
    """
      Function to handle the parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    self.name = paramInput.parameterValues['name']## Figure out field, data, X,Y from features etc..
    for child in paramInput.subparts:
      self.sparseSensingGoal = child.parameterValues['subType']
      self.nSensors = child.findFirst('nSensors').value
      self.nModes = child.findFirst('nModes').value
      self.basis = child.findFirst('basis').value
      self.sensingFeatures = child.findFirst('features').value
      self.sensingStateVariable = child.findFirst('measuredState').value## This is Field
      if self.sparseSensingGoal == 'classification':
        self.sensingLabels = child.findFirst('labels').value
      self.ConstrainedRegions = child.findFirst('ConstrainedRegions')
      if self.ConstrainedRegions is not None:
        self._ConstrainedRegionsType = self.ConstrainedRegions.parameterValues['type']
        if self._ConstrainedRegionsType == 'Circle':
          self.centerX = self.ConstrainedRegions.findFirst('centerX').value
          self.centerY = self.ConstrainedRegions.findFirst('centerY').value
          self.radius = self.ConstrainedRegions.findFirst('radius').value
          self.loc = self.ConstrainedRegions.findFirst('loc').value
        elif self._ConstrainedRegionsType == 'Ellipse':
          self.centerX = self.ConstrainedRegions.findFirst('centerX').value
          self.centerY = self.ConstrainedRegions.findFirst('centerY').value
          self.width = self.ConstrainedRegions.findFirst('width').value
          self.height = self.ConstrainedRegions.findFirst('height').value
          self.angle = self.ConstrainedRegions.findFirst('angle').value
          self.loc = self.ConstrainedRegions.findFirst('loc').value
        elif self._ConstrainedRegionsType == 'Line':
          self.x1 = self.ConstrainedRegions.findFirst('x1').value
          self.x2 = self.ConstrainedRegions.findFirst('x2').value
          self.y1 = self.ConstrainedRegions.findFirst('y1').value
          self.y2 = self.ConstrainedRegions.findFirst('y2').value
        elif self._ConstrainedRegionsType == 'Parabola':
          self.h = self.ConstrainedRegions.findFirst('h').value
          self.k = self.ConstrainedRegions.findFirst('k').value
          self.a = self.ConstrainedRegions.findFirst('a').value
          self.loc = self.ConstrainedRegions.findFirst('loc').value
        elif self._ConstrainedRegionsType == 'Polygon':
          self.loc = self.ConstrainedRegions.findFirst('loc').value
          self.xyCoords = self.ConstrainedRegions.findFirst('xyCoords').value
        # elif self._ConstrainedRregionType.lower() == 'UserDefinedConstraint'
      else:
        self.ConstrainedRegions = None
        self._ConstrainedRegionsType = None

      if child.findFirst('optimizer') is not None:
        self.optimizer = child.findFirst('optimizer').value
        if self.optimizer == 'GQR':
            self.nConstSensors = child.findFirst('nConstSensors').value
            self.constraintOption = child.findFirst('constraintOption').value
      else:
        self.optimizer = None
      if child.findFirst('classifier') is not None:
        self.classifier = child.findFirst('classifier').value
      else:
        self.classifier = None
      if child.findFirst('seed') is not None:
        self.seed = child.findFirst('seed').value
      else:
        self.seed = None
      if child.parameterValues['subType'] not in self.goalsDict.keys():
        self.raiseAnError(IOError, '{} is not a recognized option, allowed options are {}'.format(child.getName(),self.goalsDict.keys()))
    if self.sparseSensingGoal == 'classification':
      _, notFound = paramInput.subparts[0].findNodesAndExtractValues(['nModes','nSensors','features','measuredState','labels'])
    else:
      _, notFound = paramInput.subparts[0].findNodesAndExtractValues(['nModes','nSensors','features','measuredState'])
    # notFound must be empty
    assert not notFound, "Unexpected nodes in _handleInput"

  def run(self,inputIn):
    """
      This method executes the postprocessor action. In this case, it finds the optimal sensor locations to achieve a prescribed goal
      (i.e., reconstruction of a certain response of interest, or classify between data different scenarios)
      @ In, inputIn, dict, dictionaries which contains the data inside the input DataObjects
      @ Out, outputDic, dict, dictionary which contains the data to be collected by output DataObject
    """
    _, _, inputDS = inputIn['Data'][0]
    ## identify features
    self.features = self.sensingFeatures
    # don't keep the pivot parameter in the feature space
    if self.pivotParameter in self.features:
      self.features.remove(self.pivotParameter)
    if self.basis.lower() == 'svd':
      basis=ps.basis.SVD(n_basis_modes=self.nModes)
    elif self.basis.lower() == 'identity':
      basis=ps.basis.Identity(n_basis_modes=self.nModes)
    elif self.basis.lower() == 'randomprojection':
      basis=ps.basis.RandomProjection(n_basis_modes=self.nModes)
    else:
      self.raiseAnError(IOError, 'basis are not recognized')
    features = {}
    for var in self.sensingFeatures:
      features[var] = np.atleast_1d(inputDS[var].data)
    nSamples,nfeatures = np.shape(features[self.sensingFeatures[0]])
    dataframe = inputDS.to_dataframe()
    dataframe = dataframe.loc[0]
    dataframe = dataframe.reset_index()
    dataframe = dataframe.drop('index', axis=1)
    data = inputDS[self.sensingStateVariable].data
    assert np.shape(data) == (nSamples,nfeatures)
    allSensors = np.array(range(0, data.shape[1]))## Data must be [n_samples, n_features]
    if self.sparseSensingGoal == 'reconstruction':
      if self.optimizer == 'GQR':
        if self._ConstrainedRegionsType == 'Circle':### As of now : Y_axis = self.sensingFeatures[1] , X_axis = self.sensingFeatures[0] as of now dependent on order of input from the user (Can be better/need to fix)
          circle = ps.utils._constraints.Circle(center_x = self.centerX, center_y = self.centerY, radius = self.radius, loc = self.loc, data = dataframe, Y_axis = self.sensingFeatures[1] , X_axis = self.sensingFeatures[0] , Field = self.sensingStateVariable)
          idxConstrained, rank = circle.get_constraint_indices(all_sensors = allSensors, info=dataframe)
        elif self._ConstrainedRegionsType == 'Ellipse':
          ellipse = ps.utils._constraints.Ellipse(center_x = self.centerX, center_y = self.centerY, width = self.width, height = self.height, angle = self.angle, loc = self.loc, data = dataframe, Y_axis = self.sensingFeatures[1] , X_axis = self.sensingFeatures[0] , Field = self.sensingStateVariable)
          idxConstrained, rank = ellipse.get_constraint_indices(all_sensors = allSensors, info=dataframe)
        elif self._ConstrainedRegionsType == 'Line':
          line = ps.utils._constraints.Line( x1 = self.x1, x2 = self.x2, y1 = self.y1, y2 = self.y2, data = dataframe, Y_axis = self.sensingFeatures[1] , X_axis = self.sensingFeatures[0] , Field = self.sensingStateVariable)
          idxConstrained, rank = line.get_constraint_indices(all_sensors = allSensors, info=dataframe)
        elif self._ConstrainedRegionsType == 'Parabola':
          parabola = ps.utils._constraints.Parabola( h = self.h, k = self.k, a = self.a , loc = self.loc , data = dataframe, Y_axis = self.sensingFeatures[1] , X_axis = self.sensingFeatures[0] , Field = self.sensingStateVariable)
          idxConstrained, rank = parabola.get_constraint_indices(all_sensors = allSensors, info=dataframe)
        elif self._ConstrainedRegionsType == 'Polygon':
          polygon = ps.utils._constraints.Polygon( xy_coords = self.xyCoords,data = dataframe, Y_axis = self.sensingFeatures[1] , X_axis = self.sensingFeatures[0] , Field = self.sensingStateVariable)
          idxConstrained, rank = polygon.get_constraint_indices(all_sensors = allSensors, info=dataframe)
        else:
          self.raiseAnError(IOError, 'Shape is not recognized!. Currently, only Circle, Line, Polygon, Parabola, Ellipse constraint regions are implemented')
    # reconstruction, binary classification, multiclass classification or anomaly detection
        optimizer_kwargs = {'idx_constrained': idxConstrained, 'n_sensors': self.nSensors, 'n_const_sensors': self.nConstSensors,'all_sensors': allSensors, 'constraint_option': self.constraintOption}
        optimizer = ps.optimizers.GQR()
        model = ps.SSPOR(basis = basis, optimizer = optimizer, n_sensors = self.nSensors)
        if self.seed is not None:
          model.fit(data, seed=self.seed, **optimizer_kwargs)
        else:
          model.fit(data, **optimizer_kwargs)

      elif self.optimizer == 'QR':
        optimizer = ps.optimizers.QR()
        model = ps.SSPOR(basis=basis, n_sensors=self.nSensors, optimizer=optimizer)
        if self.seed is not None:
          model.fit(data, seed=self.seed)
        else:
          model.fit(data)

      else:
        self.raiseAnError(IOError, 'optimizer {} not implemented!!!'.format(self.optimizer))

    elif self.sparseSensingGoal == 'classification':
      labels = inputDS[self.sensingLabels].data[:,0]
      if self.classifier == None or self.classifier.lower() == 'lda':
        classifier = ps.classification._sspoc.LinearDiscriminantAnalysis()
        model = ps.SSPOC(basis=basis, n_sensors=self.nSensors, classifier=classifier)
        model.fit(data,y=labels)
      else:
        self.raiseAnError(IOError, 'classifier is not recognized!. Currently, only LDA classifier is implemented')

    else:
      self.raiseAnError(IOError, 'Goal is not recognized!. Currently, only regression and classification are the accepted goals')


    selectedSensors = model.get_selected_sensors()
    coords = {'sensor':np.arange(1,len(selectedSensors)+1)}

    sensorData = {}
    for var in self.sensingFeatures:
      sensorData[var] = ('sensor', inputDS[var][0,selectedSensors].data)
    outDS = xr.Dataset(data_vars=sensorData, coords=coords)
    ## PLEASE READ: For developers: this is really important, currently,
    # you have to manually add RAVEN_sample_ID to the dims if you are using xarrays
    outDS = outDS.expand_dims(self.sampleTag)
    outDS[self.sampleTag] = [0]
    return outDS