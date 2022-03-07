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
  Created on Jun 30, 2021

  @author: wangc
  (Error-Correcting) Output-Code multiclass strategy classifer
"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class OutputCodeClassifier(ScikitLearnBase):
  """
    (Error-Correcting) Output-Code multiclass strategy classifer
  """
  info = {'problemtype':'classification', 'normalize':False}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    import sklearn
    import sklearn.multiclass
    self.model = sklearn.multiclass.OutputCodeClassifier

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.description = r"""The \xmlNode{OutputCodeClassifier} (\textit{(Error-Correcting) Output-Code multiclass strategy})
                        Output-code based strategies consist in representing each class with a binary code (an array of
                        0s and 1s). At fitting time, one binary classifier per bit in the code book is fitted. At
                        prediction time, the classifiers are used to project new points in the class space and the class
                        closest to the points is chosen. The main advantage of these strategies is that the number of
                        classifiers used can be controlled by the user, either for compressing the model
                        (0 < code\_size < 1) or for making the model more robust to errors (code\_size > 1). See the
                        documentation for more details.
                        \zNormalizationNotPerformed{OutputCodeClassifier}
                        """
    estimatorInput = InputData.assemblyInputFactory("estimator", contentType=InputTypes.StringType,
                                                 descr=r"""name of a ROM that can be used as an estimator""", default='no-default')
    #TODO: Add more inputspecs for estimator
    specs.addSub(estimatorInput)
    specs.addSub(InputData.parameterInputFactory("code_size", contentType=InputTypes.FloatType,
                                                 descr=r"""Percentage of the number of classes to be used to create
                                                 the code book. A number between 0 and 1 will require fewer classifiers
                                                 than one-vs-the-rest. A number greater than 1 will require more classifiers
                                                 than one-vs-the-rest.""", default=1.5))
    specs.addSub(InputData.parameterInputFactory("random_state", contentType=InputTypes.IntegerType,
                                                 descr=r"""The generator used to initialize the codebook. Pass an int
                                                 for reproducible output across multiple function calls. """, default=None))
    specs.addSub(InputData.parameterInputFactory("n_jobs", contentType=InputTypes.IntegerType,
                                                 descr=r"""TThe number of jobs to use for the computation: the n\_classes one-vs-rest
                                                 problems are computed in parallel. None means 1 unless in a joblib.parallel\_backend
                                                 context. -1 means using all processors. See Glossary for more details.""", default=None))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['code_size', 'random_state', 'n_jobs'])
    # notFound must be empty
    assert(not notFound)
    self.settings = settings

  def setEstimator(self, estimatorList):
    """
      Initialization method
      @ In, estimatorList, list of ROM instances/estimators used by ROM
      @ Out, None
    """
    if len(estimatorList) != 1:
      self.raiseAWarning('ROM', self.name, 'can only accept one estimator, but multiple estimators are provided!',
                          'Only the first one will be used, i.e.,', estimator.name)
    estimator = estimatorList[0]
    if estimator._interfaceROM.multioutputWrapper:
      sklEstimator = estimator._interfaceROM.model.get_params()['estimator']
    else:
      sklEstimator = estimator._interfaceROM.model
    if not callable(getattr(sklEstimator, "fit", None)):
      self.raiseAnError(IOError, 'estimator:', estimator.name, 'can not be used! Please change to a different estimator')
    ###FIXME: We may need to check 'predict_proba' inside ROM class
    # elif not callable(getattr(sklEstimator, "predict_proba", None)) and not callable(getattr(sklEstimator, "decision_function", None)):
    #   self.raiseAnError(IOError, 'estimator:', estimator.name, 'can not be used! Please change to a different estimator')
    else:
      self.raiseADebug('A valid estimator', estimator.name, 'is provided!')
    settings = {'estimator':sklEstimator}
    self.settings.update(settings)
    self.initializeModel(self.settings)
