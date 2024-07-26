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
  Hankel Alternative View of Koopman (HAVOK) model

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
from ...utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class HAVOK(DMDBase):
  """
    Hankel Alternative View of Koopman (HAVOK) (Parametric)
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
    self._dmdBase = None # {} # HAVOK
    self.fitArguments = {'t': 1}

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(HAVOK, cls).getInputSpecification()

    specs.description = r"""The \xmlString{HAVOK} ROM (Hankel Alternative View of Koopman - HAVOK) aimed to construct a time-dependent (or any other monotonic
    variable) surrogate model based on Hankel Alternative View of Koopman model
    This surrogate is aimed to perform a ``dimensionality reduction regression'', where, given time
    series (or any monotonic-dependent variable) of data, a set of modes each of which is associated
    with a fixed oscillation frequency and decay/growth rate is computed
    in order to represent the data-set.
    In order to use this Reduced Order Model, the \xmlNode{ROM} attribute
    \xmlAttr{subType} needs to be set equal to \xmlString{DMD}.
    \\
    Once the ROM  is trained (\textbf{Step} \xmlNode{RomTrainer}), its parameters/coefficients can be exported into an XML file
    via an \xmlNode{OutStream} of type \xmlAttr{Print}. The following variable/parameters  can be exported (i.e. \xmlNode{what} node
    in \xmlNode{OutStream} of type \xmlAttr{Print}):
    \begin{itemize}
      \item \xmlNode{svd\_rank}, see XML input specifications below
      \item \xmlNode{delays}, see XML input specifications below
      \item \xmlNode{lag}, see XML input specifications below
      \item \xmlNode{num\_chaos}, see XML input specifications below
      \item \xmlNode{structured}, see XML input specifications below
      \item \xmlNode{lstsq}, see XML input specifications below
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
                                                 If $0.0 < svd_rank < 1.0$, this parameter represents the energy level.The value is used to compute the rank such
                                                   as computed rank is the number of the biggest singular values needed to reach the energy identified by
                                                   \xmlNode{svd_rank}.
                                                 """, default=0))
    specs.addSub(InputData.parameterInputFactory("delays", contentType=InputTypes.IntegerType,
                                                 descr=r"""The number of consecutive time-shifted copies of the data to use when building Hankel matrices.
                                                 Note that if examining an n-dimensional data set, this means that the resulting Hankel matrix
                                                 will contain $n * delays$ rows""", default=10))
    specs.addSub(InputData.parameterInputFactory("lag", contentType=InputTypes.IntegerType,
                                                 descr=r"""The number of time steps between each time-shifted copy of data in the Hankel matrix.
                                                 This means that each row of the Hankel matrix will be separated by a time-step of $dt * lag$.""", default=1))
    specs.addSub(InputData.parameterInputFactory("num_chaos", contentType=InputTypes.IntegerType,
                                                 descr=r"""The number of forcing terms to use in the HAVOK model.""", default=1))
    specs.addSub(InputData.parameterInputFactory("structured", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether to perform standard HAVOK or structured HAVOK (sHAVOK).
                                                 If True, sHAVOK is performed, otherwise HAVOK is performed.""", default=False))
    specs.addSub(InputData.parameterInputFactory("lstsq", contentType=InputTypes.BoolType,
                                                 descr=r"""Method used for computing the HAVOK operator.
                                                 If True, least-squares is used, otherwise the pseudo- inverse is used.""",
                                                 default=True))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    import pydmd
    from pydmd import HAVOK
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['svd_rank', 'delays','lag', 'num_chaos', 'structured', 'lstsq'])
    # notFound must be empty
    assert(not notFound)
    # -1 no truncation, 0 optimal rank is computed, >1 truncation rank
    # if 0.0 < float < 1.0, computed rank is the number of the biggest sv needed to reach the energy identified by this float value
    self.dmdParams['svd_rank'] = settings.get('svd_rank')
    # The number of consecutive time-shifted copies of the data to use when building Hankel matrices.
    self.dmdParams['delays'] = settings.get('delays')
    # the number of time steps between each time-shifted copy of data in the Hankel matrix
    self.dmdParams['lag'      ] = settings.get('lag')
    # The number of forcing terms to use in the HAVOK model.
    self.dmdParams['num_chaos'] = settings.get('num_chaos')
    # Whether to perform standard HAVOK or structured HAVOK (sHAVOK)
    self.dmdParams['structured'] = settings.get('structured')
    # Method used for computing the HAVOK operator.
    self.dmdParams['lstsq'] = settings.get('lstsq')

    # for target
    #for target in  set(self.target) - set(self.pivotID):
    #  self._dmdBase[target] = HankelDMD
    self._dmdBase = HAVOK
    # intialize the model
    self.initializeModel(settings)

  def _preFitModifications(self):
    """
      Method to modify parameters and populate fit argument before fitting
      @ In, None
      @ Out, None
    """
    self.fitArguments['t'] = self.pivotValues.flatten()
    if len(self.fitArguments['t']) < self.dmdParams['delays']:
      self.raiseAWarning(f'In ROM {self.name} "delays" argument is set to {delays} but the # ts is {len(self.fitArguments["t"])}. '
                         'Modifying parameter accordingly!')
    self.dmdParams['delays'] = min(self.dmdParams['delays'], len(self.fitArguments['t']))
