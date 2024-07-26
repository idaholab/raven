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
  Optimized DMD and Bagging, Optimized DMD

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

class BOPDMD(DMDBase):
  """
    Optimized DMD and Bagging, Optimized DMD (Parametric)
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
    self._dmdBase = None # {} # BOPDMD
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
    specs = super(BOPDMD, cls).getInputSpecification()

    specs.description = r"""The \xmlString{BOPDMD} ROM (Optimized DMD and Bagging, Optimized DMD) aimed to construct a time-dependent (or any other monotonic
    variable) surrogate model based on FOptimized DMD and Bagging, Optimized DMD.
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
      \item \xmlNode{opt}, see XML input specifications below
      \item \xmlNode{exact}, see XML input specifications below
      \item \xmlNode{rescale\_mode}, see XML input specifications below
      \item \xmlNode{forward\_backward}, see XML input specifications below
      \item \xmlNode{d}, see XML input specifications below
      \item \xmlNode{eig\_sort}, see XML input specifications below
      \item \xmlNode{reconstruction\_method}, see XML input specifications below
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
                                                 \item \textit{>1}, this rank is going to be used for the truncation
                                                 \end{itemize}
                                                 If $0.0 < svd_rank < 1.0$, this parameter represents the energy level.The value is used to compute the rank such
                                                   as computed rank is the number of the biggest singular values needed to reach the energy identified by
                                                   \xmlNode{svd_rank}.
                                                 """, default=0))
    specs.addSub(InputData.parameterInputFactory("compute_A", contentType=InputTypes.BoolType,
                                                 descr=r"""Flag that determines whether or not to compute the full Koopman operator A.
                                                 Default is False, do not compute the full operator. Note that the full operator
                                                 is potentially prohibitively expensive to compute.""", default=False))
    specs.addSub(InputData.parameterInputFactory("use_proj", contentType=InputTypes.BoolType,
                                                 descr=r"""Flag that determines the type of computation to perform. If True, fit input
                                                 data projected onto the first svd_rank POD modes or columns of proj_basis if provided.
                                                 If False, fit the full input data. Default is True, fit projected data.""", default=True))
    
    specs.addSub(InputData.parameterInputFactory("num_trials", contentType=InputTypes.IntegerType,
                                                 descr=r"""Number of BOP-DMD trials to perform. If num_trials is a positive integer,
                                                 num\_trials BOP-DMD trials are performed. Otherwise, standard optimized dmd is performed""", default=0))    
    specs.addSub(InputData.parameterInputFactory("trial_size", contentType=InputTypes.FloatOrIntType,
                                                 descr=r"""Size of the randomly selected subset of observations to use for each trial of bagged optimized dmd (BOP-DMD).
                                                 Available options are:
                                                 \begin{itemize}
                                                   \item \textit{>1}, trial\_size many observations will be used per trial.
                                                   \item $0.0 < trial\_size < 1.0$, $int(trial\_size * m)$ many observations will be used per trial,
                                                       where $m$ denotes the total number of data points observed.
                                                 \end{itemize}
                                                 """, default=0.6))    
    specs.addSub(InputData.parameterInputFactory("eig_sort", contentType=InputTypes.makeEnumType("eig_sort", "SortedType",
                                                                                                        ["real", "abs", "imag", "auto"]),
                                                 descr=r"""Method used to sort eigenvalues (and modes accordingly) when performing BOP-DMD. Available options are:
                                                 \begin{itemize}
                                                   \item \textit{real}, eigenvalues will be sorted by real part and then by imaginary part to break ties.
                                                   \item  \textit{imag}, eigenvalues will be sorted by imaginary part and then by real part to break ties.
                                                   \item  \textit{abs}, eigenvalues will be sorted by magnitude.
                                                   \item  \textit{auto}, one of the previously-mentioned sorting methods is chosen depending on eigenvalue variance.  
                                                 \end{itemize}
                                                 """, default="auto"))
    specs.addSub(InputData.parameterInputFactory("eig_constraints", contentType=InputTypes.makeEnumType("eig_constraints", "EigenConstraintType",
                                                                                                        ["stable", "imag", "conjugate_pairs", None]),
                                                 descr=r"""Set containing desired DMD operator eigenvalue constraints.. Available options are:
                                                 \begin{itemize}
                                                   \item \textit{stable}, constrains eigenvalues to the left half of the complex plane.
                                                   \item  \textit{imag}, constrains eigenvalues to the imaginary axis.
                                                   \item  \textit{conjugate_pairs}, enforces that eigenvalues are always present with their complex conjugate.
                                                 \end{itemize}
                                                 """, default=None))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    import pydmd
    from pydmd import BOPDMD
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['svd_rank', 'compute_A', 'use_proj', 'num_trials', 'trial_size',
                                                               'eig_sort', 'eig_constraints'])
    # notFound must be empty
    assert(not notFound)
    # -1 no truncation, 0 optimal rank is computed, >1 truncation rank
    # if 0.0 < float < 1.0, computed rank is the number of the biggest sv needed to reach the energy identified by this float value
    self.dmdParams['svd_rank'] = settings.get('svd_rank')
    # compute full A matrix
    self.dmdParams['compute_A'] = settings.get('compute_A')
    # Type of projection to perform
    self.dmdParams['use_proj'] = settings.get('use_proj')    
    # Number of trials
    self.dmdParams['num_trials'] = settings.get('num_trials')
    # Trial size
    self.dmdParams['trial_size'] = settings.get('trial_size')
    # Eigen value constraints to apply
    self.dmdParams['eig_constraints'] = set([settings.get('eig_constraints')])
    # Sorted eigs
    self.dmdParams['eig_sort'] = settings.get('eig_sort')

    self._dmdBase = BOPDMD
    # intialize the model
    self.initializeModel(settings)
    
  def _preFitModifications(self):
    """
      Method to modify parameters and populate fit argument before fitting
      @ In, None
      @ Out, None
    """
    self.fitArguments['t'] = self.pivotValues.flatten()