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
import numpy as np
import xarray as xr

from .PostProcessorReadyInterface import PostProcessorReadyInterface
from ...utils import InputData, InputTypes

class SparseSensing(PostProcessorReadyInterface):
  """
    This Postprocessor class finds the optimal locations of sparse sensors for both classification and reconstruction problems.
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
    features = InputData.parameterInputFactory("features", contentType=InputTypes.StringListType, ##Change with variable X,Y
                                                printPriority=108,
                                                descr=r"""Features/inputs of the data model""")
    goal.addSub(features)
    xValue = InputData.parameterInputFactory("xValue", contentType=InputTypes.StringListType,
                                                printPriority=108,
                                                descr=r"""Variable plotted on the X-axis of the data model""")
    goal.addSub(xValue)
    yValue = InputData.parameterInputFactory("yValue", contentType=InputTypes.StringListType,
                                                printPriority=108,
                                                descr=r"""Variable plotted on the Y-axis of the data model""")  ##Do we want another Z-value
    goal.addSub(yValue)
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
    ConstrainedRegions = InputData.parameterInputFactory("ConstrainedRegions", contentType=InputTypes.StringType,
                                                printPriority=108,
                                                descr=r"""The shape of region we want to constrain""")
    goal.addSub(ConstrainedRegions)
    ConstrainedRegions.addParam("type", InputTypes.makeEnumType("Constraint","ConstraintType",['Circle','Ellipse','Line','Polygon','Parabola']), True,
                       descr="type of Constrained Region shape the user wants- (Constraint can be a circle, parabola, ellipse, line, rectangle, square or user defined constraint too)")    ## Defining 'type' and making it Enum type.
    optimizer = InputData.parameterInputFactory("optimizer", contentType=InputTypes.makeEnumType("optimizer","optimizer type",['QR']),
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
    center_x = InputData.parameterInputFactory("center_x", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""The x co-ordinate of center of circle,ellipse""")
    goal.addSub(center_x)
    center_y = InputData.parameterInputFactory("center_y", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""The y co-ordinate of center of circle,ellipse""")
    goal.addSub(center_y)
    radius = InputData.parameterInputFactory("radius", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""The radius of circle,ellipse""")
    goal.addSub(radius)
    loc = InputData.parameterInputFactory("loc", contentType=InputTypes.makeEnumType("Constraint","Constraint Locations",['in','out']),
                                                           printPriority=108,
                                                           descr=r"""Whether the constraint region is inside or outside the shape defined as constraint""")
    goal.addSub(loc)
    width = InputData.parameterInputFactory("width", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""Width of the ellipse""")
    goal.addSub(width)
    height = InputData.parameterInputFactory("height", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""Height of the ellipse""")
    goal.addSub(height)
    angle = InputData.parameterInputFactory("angle", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""Angle of rotation of the ellipse""")
    goal.addSub(angle)
    x1 = InputData.parameterInputFactory("x1", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""X co-ordinate of one of the points that defines the line""")
    goal.addSub(x1)
    x2 = InputData.parameterInputFactory("x2", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""X co-ordinate of the other point that defines the line""")
    goal.addSub(x2)
    y1 = InputData.parameterInputFactory("y1", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""Y co-ordinate of one of the points that defines the line""")
    goal.addSub(y1)
    y2 = InputData.parameterInputFactory("y2", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""Y co-ordinate of the other point that defines the line""")
    goal.addSub(y2)
    h = InputData.parameterInputFactory("h", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""X coordinate of the vertex of the parabola we want to be constrained""")
    goal.addSub(h)
    k = InputData.parameterInputFactory("k", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r""" Y coordinate of the vertex of the parabola we want to be constrained""")
    goal.addSub(k)
    a = InputData.parameterInputFactory("a", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""X coordinate of the focus of the parabola""")
    goal.addSub(a)
    xy_coords = InputData.parameterInputFactory("xy_coords", contentType=InputTypes.IntegerListType,   ##What is the type for an array?
                                                           printPriority=108,
                                                           descr=r"""an array consisting of tuples for (x,y) coordinates of points of the Polygon where N = No. of sides of the polygon""")
    goal.addSub(xy_coords)
    inputSpecification.addSub(goal)
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
    self.center_x = None                                     # The x co-ordinate of center of circle,ellipse
    self.center_y = None                                     # The y co-ordinate of center of circle,ellipse
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
    self.xy_coords = None                                    # an array consisting of tuples for (x,y) coordinates of points of the Polygon where N = No. of sides of the polygon



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
    self.name = paramInput.parameterValues['name']  ## Figure out field, data, X,Y from features etc..
    for child in paramInput.subparts:
      self.sparseSensingGoal = child.parameterValues['subType']
      self.nSensors = child.findFirst('nSensors').value
      self.nModes = child.findFirst('nModes').value
      self.basis = child.findFirst('basis').value
      self.sensingFeatures = child.findFirst('features').value
      self.sensingStateVariable = child.findFirst('measuredState').value  ## This is Field
      if self.sparseSensingGoal == 'classification':
        self.sensingLabels = child.findFirst('labels').value
      if child.findFirst('ConstrainedRegion') is not None:
        self.ConstrainedRegion = child.findFirst('ConstrainedRegion').value
        self._ConstrainedRegionType = self.ConstrainedRegion.parameterValues['type']  ##Check for is not None
        if self._ConstrainedRegionType.lower() == 'circle':
          self.center_x = child.findFirst('center_x').value
          self.center_y = child.findFirst('center_y').value
          self.radius = child.findFirst('radius').value
          self.loc = child.findFirst('loc').value
        elif self._ConstrainedRegionType.lower() == 'ellipse':
          self.center_x = child.findFirst('center_x').value
          self.center_y = child.findFirst('center_y').value
          self.width = child.findFirst('width').value
          self.height = child.findFirst('height').value
          self.angle = child.findFirst('angle').value
          self.loc = child.findFirst('loc').value
        elif self._ConstrainedRegionType.lower() == 'line':
          self.x1 = child.findFirst('x1').value
          self.x2 = child.findFirst('x2').value
          self.y1 = child.findFirst('y1').value
          self.y2 = child.findFirst('y2').value
        elif self._ConstrainedRegionType.lower() == 'parabola':
          self.h = child.findFirst('h').value
          self.k = child.findFirst('k').value
          self.a = child.findFirst('a').value
          self.loc = child.findFirst('loc').value
        elif self._ConstrainedRegionType.lower() == 'polygon':
          self.xy_coords = child.findFirst('xy_coords').value
          self.loc = child.findFirst('loc').value
        # elif self._ConstrainedRregionType.lower() == 'UserDefinedConstraint'

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

    if self.sparseSensingGoal == 'reconstruction':
      if self.optimizer.lower() == 'qr':
        optimizer = ps.optimizers.QR()
      ## TODO: Add GQR for constrained optimization
      elif self.optimizer.lower() == 'gqr':
        optimizer = ps.optimizers.GQR()
      ## TODO: Add CCQR for cost constrained optimization
      elif self.optimizer.lower() == 'ccqr':
        optimizer = ps.optimizer.CCQR()
      else:
        self.raiseAnError(IOError, 'optimizer {} not implemented!!!'.format(self.optimizer))
    elif  self.sparseSensingGoal == 'classification':
      if self.classifier == None or self.classifier.lower() == 'lda':
        classifier = ps.classification._sspoc.LinearDiscriminantAnalysis()
      else:
        self.raiseAnError(IOError, 'classifier is not recognized!. Currently, only LDA classifier is implemented')
    else:
      self.raiseAnError(IOError, 'Goal is not recognized!. Currently, only regression and classification are the accepted goals')
    ### As of now : Y_axis = self.sensingFeatures[1] , X_axis = self.sensingFeatures[0] as of now dependent on order of input from the user (Can be better/need to fix)
    data = inputDS[self.sensingStateVariable].data  ## Moved data up so as to pass into instance of constraint class
    allSensors = range(0, data.shape[1]) ## Data must be [n_samples, n_features]
    if self._ConstrainedRegionType.lower() == 'circle':
      circle = ps.utils._constraints.Circle(center_x = self.center_x, center_y = self.center_y, radius = self.radius, loc = self.loc, data = data, Y_axis = self.sensingFeatures[1] , X_axis = self.sensingFeatures[0] , Field = self.sensingStateVariable)
      idxConstrained, rank = circle.get_constraint_indices(all_sensors = allSensors, info=data)
    if self._ConstrainedRegionType.lower() == 'ellipse':
      ellipse = ps.utils._constraints.Ellipse(center_x = self.center_x, center_y = self.center_y, width = self.width, height = self.height, angle = self.angle, loc = self.loc, data = data, Y_axis = self.sensingFeatures[1] , X_axis = self.sensingFeatures[0] , Field = self.sensingStateVariable)
      idxConstrained, rank = ellipse.get_constraint_indices(all_sensors = allSensors, info=data)
    if self._ConstrainedRegionType.lower() == 'line':
      line = ps.utils._constraints.Line( x1 = self.x1, x2 = self.x2, y1 = self.y1, y2 = self.y2, data = data, Y_axis = self.sensingFeatures[1] , X_axis = self.sensingFeatures[0] , Field = self.sensingStateVariable)
      idxConstrained, rank = line.get_constraint_indices(all_sensors = allSensors, info=data)
    if self._ConstrainedRegionType.lower() == 'parabola':
      parabola = ps.utils._constraints.Parabola( h = self.h, k = self.k, a = self.a , loc = self.loc , data = data, Y_axis = self.sensingFeatures[1] , X_axis = self.sensingFeatures[0] , Field = self.sensingStateVariable)
      idxConstrained, rank = parabola.get_constraint_indices(all_sensors = allSensors, info=data)
    if self._ConstrainedRegionType.lower() == 'polygon':
      polygon = ps.utils._constraints.Polygon( xy_coords = self.xy_coords,data = data, Y_axis = self.sensingFeatures[1] , X_axis = self.sensingFeatures[0] , Field = self.sensingStateVariable)
      idxConstrained, rank = polygon.get_constraint_indices(all_sensors = allSensors, info=data)
    # reconstruction, binary classification, multiclass classification or anomaly detection
    if self.sparseSensingGoal == 'reconstruction':
      if self.optimizer == 'GQR':
        optimizer_kwargs = {'idx_constrained': idxConstrained, 'n_sensors': self.n_sensors, 'n_const_sensors': self.n_const_sensors,'all_sensors': self.all_sensors, 'constraint_option': self.constraint_option}
      model = ps.SSPOR(basis=basis, n_sensors=self.nSensors, optimizer=optimizer)
    else:
      model = ps.SSPOC(basis=basis, n_sensors=self.nSensors, classifier=classifier)
    features = {}
    for var in self.sensingFeatures:
      features[var] = np.atleast_1d(inputDS[var].data)
    nSamples,nfeatures = np.shape(features[self.sensingFeatures[0]])
    ##TODO ##FIXME
    if self.sparseSensingGoal == 'classification':
      ## TODO: maybe add another variable called state variable to distinguish between target or label,
      # (target is temperature and label is whatever label like 'P_T', or '<T*' '>T*')
      # Other option is to keep label as the target and add another variable call it state variable, in the classification state will be different than target
      # Also for LDA we have to error out if number of classes is not less than number of samples
      labels = inputDS[self.sensingLabels].data[:,0]
    ## TODO: add some assertions to check the shape of the data matrix in case of steady state and time-dependent data
    assert np.shape(data) == (nSamples,nfeatures)
    if self.seed is not None and self.sparseSensingGoal == 'reconstruction':
      if self.optimizer == 'QR':
        model.fit(data, seed=self.seed)
      elif self.optimizer == 'GQR':
        model.fit(data, seed=self.seed, **optimizer_kwargs)
    elif self.sparseSensingGoal == 'reconstruction':
      if self.optimizer == 'QR':
        model.fit(data)
      elif self.optimizer == 'GQR':
        model.fit(data, **optimizer_kwargs)
    elif self.sparseSensingGoal == 'classification':
      model.fit(data,y=labels)
    else:
      raise NotImplementedError('Goal has to be either reconstruction or classification')
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
