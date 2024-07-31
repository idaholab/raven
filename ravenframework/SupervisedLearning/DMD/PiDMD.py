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

  @author: Andrea Alfonsi
  Physics-Informed Dynamic Mode Decomposition

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

class PiDMD(DMDBase):
  """
    Physics-Informed Dynamic Mode Decomposition (Parametric)
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
    self._dmdBase = None #{} # PiDMD

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(PiDMD, cls).getInputSpecification()

    specs.description = r"""The \xmlString{PiDMD} ROM aimed to construct a time-dependent (or any other monotonic
    variable) surrogate model based on the Physics Informed Dynamic Mode Decomposition
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
      \item \xmlNode{manifold}, see XML input specifications below
      \item \xmlNode{compute\_A}, see XML input specifications below
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
    specs.addSub(InputData.parameterInputFactory("opt", contentType=InputTypes.BoolType,
                                                 descr=r"""True if the amplitudes need to be computed minimizing the error
                                                  between the modes and all the time-steps or False, if only the 1st timestep only needs to be considered""",
                                                 default=False))
    specs.addSub(InputData.parameterInputFactory("manifold", contentType=InputTypes.makeEnumType("factorization", "FactorizationType",
                                                                                                        ["unitary", "uppertriangular",
                                                                                                         "lowertriangular", "diagonal",
                                                                                                         "symmetric", "skewsymmetric",
                                                                                                         "toeplitz", "hankel", "circulant",
                                                                                                         "circulant_unitary", "circulant_symmetric",
                                                                                                         "circulant_skewsymmetric", "symmetric_tridiagonal",
                                                                                                         "BC", "BCTB", "BCCB", "BCCBunitary", "BCCBsymmetric",
                                                                                                         "BCCBskewsymmetric"]),
                                                 descr=r""" the matrix manifold to restrict the full operator A to. Available options are:
                                                 \begin{itemize}
                                                   \item \textit{unitary},
                                                   \item \textit{uppertriangular},
                                                   \item \textit{lowertriangular},
                                                   \item \textit{diagonal},
                                                   \item \textit{symmetric},
                                                   \item \textit{skewsymmetric},
                                                   \item \textit{toeplitz},
                                                   \item \textit{hankel},
                                                   \item \textit{circulant},
                                                   \item \textit{circulant\_unitary},
                                                   \item \textit{circulant\_symmetric},
                                                   \item \textit{circulant\_skewsymmetric},
                                                   \item \textit{symmetric\_tridiagonal},
                                                   \item \textit{BC} (block circulant),
                                                   \item \textit{BCTB} (BC with tridiagonal blocks),
                                                   \item \textit{BCCB} (BC with circulant blocks),
                                                   \item \textit{BCCBunitary} (BCCB and unitary),
                                                   \item \textit{BCCBsymmetric} (BCCB and symmetric),
                                                   \item \textit{BCCBskewsymmetric} (BCCB and skewsymmetric).
                                                 \end{itemize}
                                                 """, default=None))
    specs.addSub(InputData.parameterInputFactory("compute_A", contentType=InputTypes.BoolType,
                                                 descr=r"""Flag that determines whether or not to compute the full Koopman operator A""",
                                                 default=False))
    specs.addSub(InputData.parameterInputFactory("manifold_opt", contentType=InputTypes.StringListType,
                                                 descr=r"""Option used to specify certain manifolds:
                                                 \begin{itemize}
                                                   \item If manifold $==$ \textit{diagonal}, \textit{manifold\_opt} may be used to specify the width of the diagonal of A:
                                                   \begin{itemize}
                                                     \item If manifold_opt is an integer $k$, A is banded, with a lower and upper bandwidth of $k-1$.
                                                     \item If manifold_opt is a tuple containing two integers $k1$ and $k2$, A is banded with
                                                           a lower bandwidth of $k1-1$ and an upper bandwidth of $k2-1$
                                                 .   \end{itemize}
                                                   \item If manifold $==$ \textit{BC},\textit{BCTB},\textit{BCCB},\textit{BCCBunitary},\textit{BCCBsymmetric},
                                                     \textit{BCCBskewsymmetric}, \textit{manifold\_opt} must be a 2D tuple that specifies the desired dimensions
                                                      of the blocks of A.
                                                 \end{itemize}

                                                 Note that all other manifolds do not use \textit{manifold\_opt}.""",
                                                 default=None))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    import pydmd
    from pydmd import PiDMD
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['svd_rank', 'tlsq_rank','compute_A', 'opt', 'manifold', 'manifold_opt'])
    # notFound must be empty
    assert(not notFound)
    # -1 no truncation, 0 optimal rank is computed, >1 truncation rank
    # if 0.0 < float < 1.0, computed rank is the number of the biggest sv needed to reach the energy identified by this float value
    self.dmdParams['svd_rank'] = settings.get('svd_rank')
    # truncation rank for total least square
    self.dmdParams['tlsq_rank'] = settings.get('tlsq_rank')
    # Compute full A operator?
    self.dmdParams['compute_A'] = settings.get('compute_A')
    # amplitudes computed minimizing the error between the mods and all the timesteps (True) or 1st timestep only (False)
    self.dmdParams['opt'] = settings.get('opt')
    # Manifold
    self.dmdParams['manifold'] = settings.get('manifold')
    if  self.dmdParams['manifold'] is None:
      self.raiseAnError(IOError, f"XML node <manifold> must be inputted for ROM type 'PiDMD' named {self.name}")
    # Manifold opt
    self.dmdParams['manifold_opt'] = settings.get('manifold_opt')

    if self.dmdParams['manifold'].startswith("BC") or self.dmdParams['manifold'] == 'diagonal':
      if self.dmdParams['manifold_opt'] is None:
        self.raiseAnError(IOError, f"XML node <manifold_opt> must be inputted for ROM type 'PiDMD' named {self.name} if"
                          f" choosen <manifold> is {self.dmdParams['manifold']}")
      if len(self.dmdParams['manifold_opt']) == 1:
        # this is an integer
        self.dmdParams['manifold_opt'] = int(self.dmdParams['manifold_opt'][0])
      else:
        # tuple
        self.dmdParams['manifold_opt'] = tuple([int(el) for el in self.dmdParams['manifold_opt']])

    self._dmdBase = PiDMD
    # intialize the model
    self.initializeModel(settings)
