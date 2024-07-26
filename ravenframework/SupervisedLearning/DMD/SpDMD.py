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

  @author: alfoa
  Sparsity-Promoting Dynamic Mode Decomposition

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

class SpDMD(DMDBase):
  """
    Sparsity-Promoting Dynamic Mode Decomposition (Parametric)
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
    self._dmdBase = None # SpDMD

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(SpDMD, cls).getInputSpecification()

    specs.description = r"""The \xmlString{SpDMD} ROM (Sparsity-Promoting Dynamic Mode Decomposition) aimed to construct a time-dependent (or any other monotonic
    variable) surrogate model based on Sparsity-Promoting Dynamic Mode Decomposition, which promotes solutions having an high number of
    amplitudes set to zero (i.e. sparse solutions). Reference: 10.1063/1.4863670.
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
      \item \xmlNode{tlsq\_rank}, see XML input specifications below
      \item \xmlNode{exact}, see XML input specifications below
      \item \xmlNode{opt}, see XML input specifications below
      \item \xmlNode{rescale\_mode}, see XML input specifications below
      \item \xmlNode{forward\_backward}, see XML input specifications below
      \item \xmlNode{sorted\_eigs}, see XML input specifications below
      \item \xmlNode{abs\_tolerance}, see XML input specifications below
      \item \xmlNode{rel\_tolerance}, see XML input specifications below
      \item \xmlNode{max\_iterations}, see XML input specifications below
      \item \xmlNode{rho}, see XML input specifications below
      \item \xmlNode{gamma}, see XML input specifications below
      \item \xmlNode{verbose}, see XML input specifications below
      \item \xmlNode{enforce\_zero}, see XML input specifications below
      \item \xmlNode{release\_memory}, see XML input specifications below
      \item \xmlNode{zero\_absolute\_tolerance}, see XML input specifications below
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
                                                   \xmlNode{energyRankSVD}. This node has always priority over  \xmlNode{rankSVD}
                                                 """, default=0))
    specs.addSub(InputData.parameterInputFactory("tlsq_rank", contentType=InputTypes.IntegerType,
                                                 descr=r"""$int > 0$ that defines the truncation rank to be used for the total
                                                  least square problem. If not inputted, no truncation is applied""", default=None))
    specs.addSub(InputData.parameterInputFactory("opt", contentType=InputTypes.FloatType,
                                                 descr=r"""True if the amplitudes need to be computed minimizing the error
                                                  between the modes and all the time-steps or False, if only the 1st timestep only needs to be considered""", default=False))
    specs.addSub(InputData.parameterInputFactory("forward_backward", contentType=InputTypes.BoolType,
                                                 descr=r"""If True, the low-rank operator is computed like in fbDMD (reference: https://arxiv.org/abs/1507.02264).
                                                 Default is False.""", default=False))
    specs.addSub(InputData.parameterInputFactory("rescale_mode", contentType=InputTypes.makeEnumType("rescale_mode", "RescaleType",
                                                                                                        ["auto", 'None']),
                                                 descr=r"""Scale Atilde as shown in 10.1016/j.jneumeth.2015.10.010 (section 2.4) before computing its eigendecomposition. None means no rescaling, ‘auto’ means automatic rescaling using singular values.
                                                 """, default=None))
    specs.addSub(InputData.parameterInputFactory("sorted_eigs", contentType=InputTypes.makeEnumType("sorted_eigs", "SortedType",
                                                                                                        ["real", "abs", 'False']),
                                                 descr=r"""Sort eigenvalues (and modes/dynamics accordingly) by magnitude if sorted_eigs=``abs'',
                                                 by real part (and then by imaginary part to break ties) if sorted_eigs=``real''.
                                                 """, default=False))
    specs.addSub(InputData.parameterInputFactory("abs_tolerance", contentType=InputTypes.FloatType,
                                                 descr=r"""Controls the convergence of ADMM. Absolute tolerance from iteration $k$ and $k-1$.""", default=1e-06))
    specs.addSub(InputData.parameterInputFactory("rel_tolerance", contentType=InputTypes.FloatType,
                                                 descr=r"""Controls the convergence of ADMM. Relative tolerance from iteration $k$ and $k-1$.""", default=0.0001))
    specs.addSub(InputData.parameterInputFactory("max_iterations", contentType=InputTypes.IntegerType,
                                                 descr=r"""The maximum number of iterations performed by ADMM, after that the algorithm is stopped.""", default=10000))
    specs.addSub(InputData.parameterInputFactory("rho", contentType=InputTypes.FloatType,
                                                 descr=r"""Controls the convergence of ADMM. For a reference on the optimal value for rho see
                                                 10.1109/TAC.2014.2354892 or 10.3182/20120914-2-US-4030.00038.""", default=1))
    specs.addSub(InputData.parameterInputFactory("gamma", contentType=InputTypes.FloatType,
                                                 descr=r"""Controls the level of “promotion” assigned to sparse solution. Increasing gamma will
                                                 result in an higher number of zero-amplitudes.""", default=10))
    specs.addSub(InputData.parameterInputFactory("verbose", contentType=InputTypes.BoolType,
                                                 descr=r"""If False, the information provided by SpDMD (like the number of iterations performed
                                                 by ADMM) are not shown.""", default=False))
    specs.addSub(InputData.parameterInputFactory("enforce_zero", contentType=InputTypes.BoolType,
                                                 descr=r"""If True the DMD amplitudes which should be set to zero
                                                 according to the solution of ADMM are manually set to 0 (since we
                                                 solve a sparse linear system to find the optimal vector of DMD amplitudes
                                                 very small terms may survive in some cases).""", default=True))
    specs.addSub(InputData.parameterInputFactory("release_memory", contentType=InputTypes.BoolType,
                                                 descr=r"""If True the intermediate matrices computed by the algorithm are deleted
                                                 after the termination of a call to train.""", default=True))
    specs.addSub(InputData.parameterInputFactory("zero_absolute_tolerance", contentType=InputTypes.FloatType,
                                                 descr=r"""Zero absolute tolerance.""", default=1e-12))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    import pydmd
    from pydmd import SpDMD
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['svd_rank', 'tlsq_rank','rescale_mode', 'sorted_eigs',
                                                                'forward_backward', 'opt',
                                                                'abs_tolerance', 'rel_tolerance','max_iterations', 'rho', 'gamma',
                                                                'verbose', 'enforce_zero', 'release_memory', 'zero_absolute_tolerance'])
    # notFound must be empty
    assert(not notFound)
    # -1 no truncation, 0 optimal rank is computed, >1 truncation rank
    # if 0.0 < float < 1.0, computed rank is the number of the biggest sv needed to reach the energy identified by this float value
    self.dmdParams['svd_rank'       ] = settings.get('svd_rank')
    # truncation rank for total least square
    self.dmdParams['tlsq_rank'] = settings.get('tlsq_rank')
    # If True, the low-rank operator is computed like in fbDMD (reference: https://arxiv.org/abs/1507.02264).
    self.dmdParams['forward_backward'] = settings.get('forward_backward')
    # abs tolerance
    self.dmdParams['abs_tolerance'] = settings.get('abs_tolerance')
    # rel tolerance
    self.dmdParams['rel_tolerance'] = settings.get('rel_tolerance')
    # max num of iterations
    self.dmdParams['max_iterations'] = settings.get('max_iterations')
    # rho
    self.dmdParams['rho'] = settings.get('rho')
    # gamma
    self.dmdParams['gamma'] = settings.get('gamma')
    # verbose?
    self.dmdParams['verbose'] = settings.get('verbose')
    # enforce_zero?
    self.dmdParams['enforce_zero'] = settings.get('enforce_zero')
    # release_memory?
    self.dmdParams['release_memory'] = settings.get('release_memory')
    # zero_absolute_tolerance
    self.dmdParams['zero_absolute_tolerance'] = settings.get('zero_absolute_tolerance')
    # Rescale mode
    self.dmdParams['rescale_mode'] = settings.get('rescale_mode')
    if self.dmdParams["rescale_mode"] is 'None':
      self.dmdParams["rescale_mode"] = None
    # Sorted eigs
    self.dmdParams['sorted_eigs'] = settings.get('sorted_eigs')
    if self.dmdParams["sorted_eigs"] is 'False':
      self.dmdParams["sorted_eigs"] = False
    # amplitudes computed minimizing the error between the mods and all the timesteps (True) or 1st timestep only (False)
    self.dmdParams['opt'] = settings.get('opt')

    self._dmdBase = SpDMD
    # intialize the model
    self.initializeModel(settings)
