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
  Created on July 21, 2024

  @author: alfoa
  Traditional Dynamic Mode Decomposition

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
from ...utils.importerUtils import importModuleLazy
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
np = importModuleLazy("numpy")
pydmd = importModuleLazy("pydmd")
ezyrb = importModuleLazy("ezyrb")
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ...SupervisedLearning.DMD import DMDBase
from ...utils import utils
from ...utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class DMD(DMDBase):
  """
    Dynamic Mode Decomposition (Parametric)
  """
  info = {'problemtype':'regression', 'normalize':False}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()

    # local model
    self._dmdBase = None #{} # DMD

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(DMD, cls).getInputSpecification()

    specs.description = r"""The \xmlString{DynamicModeDecomposition} ROM aimed to construct a time-dependent (or any other monotonic
    variable) surrogate model based on Dynamic Mode Decomposition
    This surrogate is aimed to perform a ``dimensionality reduction regression'', where, given time
    series (or any monotonic-dependent variable) of data, a set of modes each of which is associated
    with a fixed oscillation frequency and decay/growth rate is computed
    in order to represent the data-set.
    In order to use this Reduced Order Model, the \xmlNode{ROM} attribute
    \xmlAttr{subType} needs to be set equal to \xmlString{DMD}.
    \\
    Once the ROM  is trained (\textbf{Step} \xmlNode{RomTrainer}), its parameters/coefficients can be exported into an XML file
    via an \xmlNode{OutStream} of type \xmlAttr{Print}. The following variable/parameters can be exported (i.e. \xmlNode{what} node
    in \xmlNode{OutStream} of type \xmlAttr{Print}):
    \begin{itemize}
      \item \xmlNode{svd\_rank}, see XML input specifications below
      \item \xmlNode{tlsq\_rank}, see XML input specifications below
      \item \xmlNode{opt}, see XML input specifications below
      \item \xmlNode{exact}, see XML input specifications below
      \item \xmlNode{forward\_backward}, see XML input specifications below
      \item \xmlNode{tikhonov_regularization}, see XML input specifications below
      \item \xmlNode{features}, see XML input specifications below
      \item \xmlNode{timeScale}, XML node containing the array of the training time steps values
      \item \xmlNode{dmdTimeScale}, XML node containing the array of time scale in the DMD space (can be used as mapping
      between the  \xmlNode{timeScale} and \xmlNode{dmdTimeScale})
      \item \xmlNode{eigs}, XML node containing the eigenvalues (imaginary and real part)
      \item \xmlNode{amplitudes}, XML node containing the amplitudes (imaginary and real part)
      \item \xmlNode{modes}, XML node containing the dynamic modes (imaginary and real part)
    \end{itemize}"""

    specs.addSub(InputData.parameterInputFactory("svd_rank", contentType=InputTypes.FloatOrIntType,
                                                 descr=r"""defines the truncation rank to be used for the SVD.
                                                 Available options are:
                                                 \begin{itemize}
                                                 \item \textit{-1}, no truncation is performed
                                                 \item \textit{0}, optimal rank is internally computed
                                                 \item \textit{$>1$}, this rank is going to be used for the truncation

                                                 \end{itemize}
                                                 If $0.0 < svd\_rank < 1.0$, this parameter represents the energy level.The value is used to compute the rank such
                                                   as computed rank is the number of the biggest singular values needed to reach the energy identified by
                                                   \xmlNode{svd\_rank}.
                                                 """, default=0))
    specs.addSub(InputData.parameterInputFactory("tlsq_rank", contentType=InputTypes.IntegerType,
                                                 descr=r"""$int > 0$ that defines the truncation rank to be used for the total
                                                  least square problem. If not inputted, no truncation is applied""", default=None))
    specs.addSub(InputData.parameterInputFactory("exact", contentType=InputTypes.BoolType,
                                                 descr=r"""True if the exact modes need to be computed (eigenvalues and
                                                 eigenvectors),   otherwise the projected ones (using the left-singular matrix after SVD).""", default=True))
    specs.addSub(InputData.parameterInputFactory("forward_backward", contentType=InputTypes.BoolType,
                                                 descr=r"""If True, the low-rank operator is computed like in fbDMD (reference: https://arxiv.org/abs/1507.02264).
                                                 Default is False.""", default=False))
    specs.addSub(InputData.parameterInputFactory("tikhonov_regularization", contentType=InputTypes.FloatOrIntType,
                                                 descr=r"""Tikhonov parameter for the regularization.
                                                 If `None`, no regularization is applied, if `float`, it is used as the
                                                 $`\lambda`$ tikhonov parameter.""", default=None))

    specs.addSub(InputData.parameterInputFactory("opt", contentType=InputTypes.BoolType,
                                                 descr=r"""True if the amplitudes need to be computed minimizing the error
                                                  between the modes and all the time-steps or False, if only the 1st timestep only needs to be considered""", default=False))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    import pydmd
    from pydmd import DMD
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['svd_rank', 'tlsq_rank',
                                                               'exact','forward_backward','tikhonov_regularization', 'opt'])
    # notFound must be empty
    assert(not notFound)
    # -1 no truncation, 0 optimal rank is computed, >1 truncation rank
    # if 0.0 < float < 1.0, computed rank is the number of the biggest sv needed to reach the energy identified by this float value
    self.dmdParams['svd_rank'       ] = settings.get('svd_rank')
    # truncation rank for total least square
    self.dmdParams['tlsq_rank' ] = settings.get('tlsq_rank')
    # True if the exact modes need to be computed (eigs and eigvs), otherwise the projected ones (using the left-singular matrix)
    self.dmdParams['exact'      ] = settings.get('exact')
    # If True, the low-rank operator is computed like in fbDMD (reference: https://arxiv.org/abs/1507.02264).
    self.dmdParams['forward_backward'    ] = settings.get('forward_backward')
    # Tikhonov parameter for the regularization.
    self.dmdParams['tikhonov_regularization'     ] = settings.get('tikhonov_regularization')
    # amplitudes computed minimizing the error between the mods and all the timesteps (True) or 1st timestep only (False)
    self.dmdParams['opt'     ] = settings.get('opt')
    # for target
    #for target in  set(self.target) - set(self.pivotID):
    #  self._dmdBase[target] = DMD
    self._dmdBase = DMD
    # intialize the model
    self.initializeModel(settings)
