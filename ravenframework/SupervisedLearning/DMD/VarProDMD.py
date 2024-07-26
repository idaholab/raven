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
  Forward/Backward Dynamic Mode Decomposition

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

class VarProDMD(DMDBase):
  """
    Variable Projection for DMD. (Parametric)
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
    self._dmdBase = None #{} # VarProDMD
    self.fitArguments = {'time': None}

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(VarProDMD, cls).getInputSpecification()

    specs.description = r"""The \xmlString{VarProDMD} ROM (Variable Projection for DMD) aimed to construct a time-dependent (or any other monotonic
    variable) surrogate model based on Variable Projection for DMD (https://epubs.siam.org/doi/abs/10.1137/M1124176).
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
      \item \xmlNode{exact}, see XML input specifications below
      \item \xmlNode{compression}, see XML input specifications below
      \item \xmlNode{sorted\_eigs}, see XML input specifications below
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
    specs.addSub(InputData.parameterInputFactory("exact", contentType=InputTypes.BoolType,
                                                 descr=r"""True if the exact modes need to be computed (eigenvalues and
                                                 eigenvectors),   otherwise the projected ones (using the left-singular matrix after SVD).""", default=False))
    specs.addSub(InputData.parameterInputFactory("compression", contentType=InputTypes.FloatType,
                                                 descr=r"""If libary compression $c = 0$, all samples are used. If $0 < c < 1$, the best fitting
                                                 $\lfloor \left(1 - c\right)m\rfloor$ samples are selected""", default=False))
    specs.addSub(InputData.parameterInputFactory("sorted_eigs", contentType=InputTypes.makeEnumType("sorted_eigs", "SortedType",
                                                                                                        ["real", "abs", 'True', 'False', 'imag']),
                                                 descr=r"""Sort eigenvalues (and modes/dynamics accordingly) method. Available options are:
                                                 \begin{itemize}
                                                  \item \textit{True}, the variance of the absolute values of the complex eigenvalues
                                                     $\left(\sqrt{\omega_i \cdot \bar{\omega}_i}\right)$, the variance absolute values
                                                     of the real parts $\left|\Re\{{\omega_i}\}\right|$ and the variance of the absolute
                                                     values of the imaginary parts $\left|\Im\{{\omega_i}\}\right|$ is computed. The
                                                     eigenvalues are then sorted according to the highest variance (from highest to lowest).
                                                  \item \textit{False}, no sorting is performed
                                                  \item \textit{real}, the eigenvalues are sorted w.r.t. the absolute values of the real
                                                    parts of the eigenvalues (from highest to lowest).
                                                  \item \textit{imag}, the eigenvalues are sorted w.r.t. the absolute values of the imaginary
                                                    parts of the eigenvalues (from highest to lowest).
                                                  \item \textit{abs}, the eigenvalues are sorted w.r.t. the magnitude of the eigenvalues
                                                    $\left(\sqrt{\omega_i \cdot \bar{\omega}_i}\right)$ (from highest to lowest)
                                                 \end{itemize}
                                                 """, default='False'))


    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    import pydmd
    from pydmd import VarProDMD
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['svd_rank', 'exact','compression', 'sorted_eigs'])
    # notFound must be empty
    assert(not notFound)
    # -1 no truncation, 0 optimal rank is computed, >1 truncation rank
    # if 0.0 < float < 1.0, computed rank is the number of the biggest sv needed to reach the energy identified by this float value
    self.dmdParams['svd_rank'] = settings.get('svd_rank')
    # True if the exact modes need to be computed (eigs and eigvs), otherwise the projected ones (using the left-singular matrix)
    self.dmdParams['exact'      ] = settings.get('exact')
    # compression
    self.dmdParams['compression'] = settings.get('compression')
    # Sorted eigs
    self.dmdParams['sorted_eigs'] = settings.get('sorted_eigs')
    if self.dmdParams['sorted_eigs'] == 'False':
      self.dmdParams['sorted_eigs'] = False
    if self.dmdParams['sorted_eigs'] == 'True':
      self.dmdParams['sorted_eigs'] = True

    self._dmdBase = VarProDMD
    # intialize the model
    self.initializeModel(settings)

  def _preFitModifications(self):
    """
      Method to modify parameters and populate fit argument before fitting
      @ In, None
      @ Out, None
    """
    self.fitArguments['time'] = self.pivotValues.flatten()
