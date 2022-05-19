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
  Created on Jan 21, 2020

  @author: alfoa, wangc
  Nu Support Vector Classifier

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class NuSVR(ScikitLearnBase):
  """
    Nu Support Vector Classifier
  """
  info = {'problemtype':'regression', 'normalize':True}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    import sklearn
    import sklearn.svm
    self.model = sklearn.svm.NuSVR

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(NuSVR, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{NuSVR} \textit{Nu-Support Vector Regression} is an Nu-Support Vector Regressor.
                            It is very similar to SVC but with the addition of the hyper-parameter Nu for controlling the
                            number of support vectors. However, unlike NuSVC, where nu replaces C,
                            here nu replaces the parameter epsilon of epsilon-SVR.
                            \zNormalizationPerformed{NuSVR}
                            """
    specs.addSub(InputData.parameterInputFactory('nu', contentType=InputTypes.FloatType,
                                                 descr=r"""An upper bound on the fraction of margin errors and
                                                 a lower bound of the fraction of support vectors. Should be in the interval $(0, 1]$.""", default=0.5))
    specs.addSub(InputData.parameterInputFactory('C', contentType=InputTypes.FloatType,
                                                 descr=r"""Regularization parameter. The strength of the regularization is inversely
                                                          proportional to C.
                                                          Must be strictly positive. The penalty is a squared l2 penalty.""", default=1.0))
    specs.addSub(InputData.parameterInputFactory("kernel", contentType=InputTypes.makeEnumType("kernel", "kernelType",['linear','poly',
                                                                                                                       'rbf','sigmoid']),
                                                 descr=r"""Specifies the kernel type to be used in the algorithm. It must be one of
                                                            ``linear'', ``poly'', ``rbf'' or ``sigmoid''.""", default='rbf'))
    specs.addSub(InputData.parameterInputFactory("degree", contentType=InputTypes.IntegerType,
                                                 descr=r"""Degree of the polynomial kernel function ('poly').Ignored by all other kernels.""",
                                                 default=3))
    specs.addSub(InputData.parameterInputFactory("gamma", contentType=InputTypes.FloatType,
                                                 descr=r"""Kernel coefficient for ``poly'', ``rbf'' or ``sigmoid''. If not input, then it uses
                                                           $1 / (n_features * X.var())$ as value of gamma""", default="scale"))
    specs.addSub(InputData.parameterInputFactory("coef0", contentType=InputTypes.FloatType,
                                                 descr=r"""Independent term in kernel function""", default=0.0))
    specs.addSub(InputData.parameterInputFactory("tol", contentType=InputTypes.FloatType,
                                                 descr=r"""Tolerance for stopping criterion""", default=1e-3))
    specs.addSub(InputData.parameterInputFactory("cache_size", contentType=InputTypes.FloatType,
                                                 descr=r"""Size of the kernel cache (in MB)""", default=200.))
    specs.addSub(InputData.parameterInputFactory("shrinking", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether to use the shrinking heuristic.""", default=True))
    specs.addSub(InputData.parameterInputFactory("max_iter", contentType=InputTypes.IntegerType,
                                                 descr=r"""Hard limit on iterations within solver.``-1'' for no limit""", default=-1))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['nu','C', 'kernel', 'degree', 'gamma', 'coef0',
                                                             'tol', 'cache_size', 'shrinking', 'max_iter'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
