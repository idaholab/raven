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
  Gaussian process regression (GPR)

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import utils
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class GaussianProcessRegressor(ScikitLearnBase):
  """
    Gaussian process regression (GPR)
  """
  info = {'problemtype':'regression', 'normalize':False}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    import sklearn
    import sklearn.gaussian_process
    self.model = sklearn.gaussian_process.GaussianProcessRegressor

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(GaussianProcessRegressor, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{GaussianProcessRegressor} is based on Algorithm 2.1 of Gaussian Processes
                         for Machine Learning (GPML) by Rasmussen and Williams. The method is a generic supervised learning
                         method primarily designed to solve regression problems.
                         The advantages of Gaussian Processes for Machine Learning are:
                         \begin{itemize}
                           \item The prediction interpolates the observations (at least for regular
                           correlation models).
                           \item The prediction is probabilistic (Gaussian) so that one can compute
                           empirical confidence intervals and exceedance probabilities that might be used
                           to refit (online fitting, adaptive fitting) the prediction in some region of
                           interest.
                           \item Versatile: different linear regression models and correlation models can
                           be specified.
                           Common models are provided, but it is also possible to specify custom models
                           provided they are stationary.
                         \end{itemize}
                         The disadvantages of Gaussian Processes for Machine Learning include:
                         \begin{itemize}
                           \item It is not sparse.
                           It uses the whole samples/features information to perform the prediction.
                           \item It loses efficiency in high dimensional spaces – namely when the
                           number of features exceeds a few dozens.
                           It might indeed give poor performance and it loses computational efficiency.
                           \item Classification is only a post-processing, meaning that one first needs
                           to solve a regression problem by providing the complete scalar float precision
                           output $y$ of the experiment one is attempting to model.
                         \end{itemize}
                         \zNormalizationNotPerformed{GaussianProcessRegressor}
                         """
    # create kernel node
    specs.addSub(InputData.parameterInputFactory("kernel", contentType=InputTypes.makeEnumType("kernel", "kernelType",['Constant', 'DotProduct', 'ExpSineSquared',
                                                                                                                   'Matern','PairwiseLinear','PairwiseAdditiveChi2','PairwiseChi2','PairwisePoly','PairwisePolynomial','PairwiseRBF','PairwiseLaplassian','PairwiseSigmoid','PairwiseCosine', 'RBF', 'RationalQuadratic', 'Custom']),
                                                 descr=r"""The kernel specifying the covariance function of the GP. If None is passed,
                                                 the kernel $Constant$ is used as default. The kernel hyperparameters are optimized during fitting and consequentially the hyperparameters are
                                                 not inputable. The following kernels are avaialable:
                                                 \begin{itemize}
                                                   \item Constant, Constant kernel: $k(x_1, x_2) = constant\_value \;\forall\; x_1, x_2$.
                                                   \item DotProduct, it is non-stationary and can be obtained from linear regression by putting $N(0, 1)$ priors on the coefficients of $x_d (d = 1, . . . , D)$
                                                                     and a prior of $N(0, \sigma_0^2)$ on the bias. The DotProduct kernel is invariant to a rotation of the coordinates about the origin, but not translations.
                                                                     It is parameterized by a parameter sigma\_0 $\sigma$ which controls the inhomogenity of the kernel.
                                                   \item ExpSineSquared, it allows one to model functions which repeat themselves exactly. It is parameterized by a length scale parameter $l>0$ and a periodicity parameter $p>0$.
                                                                         The kernel is given by $k(x_i, x_j) = \text{exp}\left(-\frac{ 2\sin^2(\pi d(x_i, x_j)/p) }{ l^ 2} \right)$ where $d(\\cdot,\\cdot)$ is the Euclidean distance.
                                                   \item Matern, is a generalization of the RBF. It has an additional parameter $\nu$ which controls the smoothness of the resulting function. The smaller $\nu$,
                                                                 the less smooth the approximated function is. As $\nu\rightarrow\infty$, the kernel becomes equivalent to the RBF kernel. When $\nu = 1/2$, the Matérn kernel becomes
                                                                 identical to the absolute exponential kernel. Important intermediate values are $\nu = 1.5$ (once differentiable functions) and $\nu = 2.5$ (twice differentiable functions).
                                                                 The kernel is given by $k(x_i, x_j) =  \frac{1}{\Gamma(\nu)2^{\nu-1}}\Bigg( \frac{\sqrt{2\nu}}{l} d(x_i , x_j ) \Bigg)^\nu K_\nu\Bigg( \frac{\sqrt{2\nu}}{l} d(x_i , x_j )\Bigg)$
                                                                 where $d(\cdot,\cdot)$ is the Euclidean distance, $K_{\nu}(\cdot)$ is a modified Bessel function and $\Gamma(\cdot)$ is the gamma function.
                                                   \item PairwiseLinear, it is a thin wrapper around the functionality of the pairwise kernels. It uses the a linear metric to calculate kernel between instances
                                                                 in a feature array. Evaluation of the gradient is not analytic but numeric and all kernels support only isotropic distances.
                                                   \item PairwiseAdditiveChi2, it is a thin wrapper around the functionality of the pairwise metrics. It uses the an additive chi squared metric to calculate kernel between instances
                                                                 in a feature array. Evaluation of the gradient is not analytic but numeric and all kernels support only isotropic distances.
                                                   \item PairwiseChi2, it is a thin wrapper around the functionality of the pairwise metrics. It uses the a chi squared metric to calculate kernel between instances
                                                                 in a feature array. Evaluation of the gradient is not analytic but numeric and all kernels support only isotropic distances.
                                                   \item PairwisePoly, it is a thin wrapper around the functionality of the pairwise metrics. It uses the a poly metric to calculate kernel between instances
                                                                 in a feature array. Evaluation of the gradient is not analytic but numeric and all kernels support only isotropic distances.
                                                   \item PairwisePolynomial, it is a thin wrapper around the functionality of the pairwise metrics. It uses the a polynomial metric to calculate kernel between instances
                                                                 in a feature array. Evaluation of the gradient is not analytic but numeric and all kernels support only isotropic distances.
                                                   \item PairwiseRbf, it is a thin wrapper around the functionality of the pairwise metrics. It uses the a rbf metric to calculate kernel between instances
                                                                 in a feature array. Evaluation of the gradient is not analytic but numeric and all kernels support only isotropic distances.
                                                   \item PairwiseLaplacian, it is a thin wrapper around the functionality of the pairwise metrics. It uses the a laplacian metric to calculate kernel between instances
                                                                 in a feature array. Evaluation of the gradient is not analytic but numeric and all kernels support only isotropic distances.
                                                   \item PairwiseSigmoid, it is a thin wrapper around the functionality of the pairwise metrics. It uses the a sigmoid metric to calculate kernel between instances
                                                                 in a feature array. Evaluation of the gradient is not analytic but numeric and all kernels support only isotropic distances.
                                                   \item PairwiseCosine, it is a thin wrapper around the functionality of the pairwise metrics. It uses the a cosine metric to calculate kernel between instances
                                                                 in a feature array. Evaluation of the gradient is not analytic but numeric and all kernels support only isotropic distances.
                                                   \item RBF, it is a stationary kernel. It is also known as the ``squared exponential'' kernel. It is parameterized by a length scale parameter $l>0$,
                                                              which can either be a scalar (isotropic variant of the kernel) or a vector with the same number of dimensions as the inputs $X$ (anisotropic variant of the kernel).
                                                              The kernel is given by $k(x_i, x_j) = \exp\left(- \frac{d(x_i, x_j)^2}{2l^2} \right)$ where $l$ is the length scale of the kernel and $d(\cdot,\cdot)$ is the Euclidean distance.
                                                   \item RationalQuadratic, it can be seen as a scale mixture (an infinite sum) of RBF kernels with different characteristic length scales. It is parameterized by a length scale parameter
                                                                            $l>0$ and a scale mixture parameter $\alpha>0$ . The kernel is given by $k(x_i, x_j) = \left(1 + \frac{d(x_i, x_j)^2 }{ 2\alpha  l^2}\right)^{-\alpha}$ where
                                                                            $d(\cdot,\cdot)$ is the Euclidean distance.
                                                   \item Custom, this option allows the user to specify any combination of the above kernels. Still under construction
                                                 \end{itemize}.""",default=None))


    specs.addSub(InputData.parameterInputFactory("alpha", contentType=InputTypes.FloatType,
                                                 descr=r"""Value added to the diagonal of the kernel matrix during fitting. This can prevent a potential numerical issue during fitting, by ensuring that the calculated
                                                           values form a positive definite matrix. It can also be interpreted as the variance of additional Gaussian measurement noise on the training observations.""",default=1e-10))
    specs.addSub(InputData.parameterInputFactory("n_restarts_optimizer", contentType=InputTypes.IntegerType,
                                                 descr=r"""The number of restarts of the optimizer for finding the kernel's parameters which maximize the log-marginal likelihood. The first run of the optimizer is performed
                                                           from the kernel's initial parameters, the remaining ones (if any) from thetas sampled log-uniform randomly from the space of allowed theta-values. If greater than
                                                           0, all bounds must be finite.""", default=0))
    specs.addSub(InputData.parameterInputFactory("normalize_y", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether the target values y are normalized, the mean and variance of the target values are set equal to 0 and 1 respectively. This is recommended for cases where zero-mean,
                                                           unit-variance priors are used.""",default=False ))
    specs.addSub(InputData.parameterInputFactory("random_state", contentType=InputTypes.IntegerType,
                                              descr=r"""Seed for the internal random number generator.""",default=None))
    specs.addSub(InputData.parameterInputFactory("optimizer", contentType=InputTypes.makeEnumType("optimizer", "optimizerType",['fmin_l_bfgs_b']),
                                                 descr=r"""Per default, the 'L-BFGS-B' algorithm from
                                                 scipy.optimize.minimize is used. If None is passed, the kernel’s
                                                 parameters are kept fixed.""",default='fmin_l_bfgs_b'))
    specs.addSub(InputData.parameterInputFactory("anisotropic", contentType=InputTypes.BoolType,
                                                 descr=r"""Boolean that determines if unique length-scales are used for each
                                                 axis of the input space (where applicable). The default is False.""",default=False))
    specs.addSub(InputData.parameterInputFactory("custom_kernel", contentType=InputTypes.StringType,
                                                 descr=r"""Defines the custom kernel constructed by the available base kernels.
                                                 Only applicable when 'kernel' is set to 'Custom'; therefore, the default is None.""",default=None))
    specs.addSub(InputData.parameterInputFactory("multioutput", contentType=InputTypes.BoolType,
                                                 descr=r"""Determines whether model will track multiple targets through the
                                                 multiouput regressor wrapper class. For most RAVEN applications, this is expected;
                                                 therefore, the default is True.""",default=True))
    return specs

  def pickKernel(self, name, lengthScaleSetting):
    """
      This method is used to pick a kernel from the iternal factory
      @ In, name, str, the kernel name
      @ In, lengthScaleSetting, bool, whether or not to make the kernel anisotropic
      @ Out, kernel, object, the kernel object
    """
    import sklearn
    # For anisotropic cases length-scale should be a vector
    if lengthScaleSetting:
      lengthScale = np.ones(len(self.features))
    else:
      lengthScale = 1

    if name.lower() == 'constant':
      kernel = sklearn.gaussian_process.kernels.ConstantKernel(constant_value_bounds=(1e-5,1e7))
    elif name.lower() == 'dotproduct':
      kernel = sklearn.gaussian_process.kernels.DotProduct()
    elif name.lower() == 'expsinesquared':
      kernel = sklearn.gaussian_process.kernels.ExpSineSquared()
    elif name.lower() == 'matern':
      kernel = sklearn.gaussian_process.kernels.Matern(length_scale=lengthScale)
    elif name.lower() == 'rbf':
      kernel = sklearn.gaussian_process.kernels.RBF(length_scale=lengthScale)
    elif name.lower() == 'rationalquadratic':
      kernel = sklearn.gaussian_process.kernels.RationalQuadratic()
    # For the pairwise kernels, slice the input string to get metric
    elif name.lower()[0:8] == 'pairwise':
      metric = name.lower()[8:]
      kernel = sklearn.gaussian_process.kernels.PairwiseKernel(metric=metric)
    else:
      self.raiseAnError(RuntimeError, f'Invalid kernel input found: {name}')
    return kernel

  def customKernel(self, kernelStatement, lengthScaleSetting):
    """
      Constructs scikit learn kernel defined by input
      @ In, kernelStatement, string, string form of desired kernel
      @ In, lengthScaleSetting, bool, whether or not to make the kernel anisotropic (where applicable)
      @ Out, kernel, sklearn kernel object
    """
    import sklearn
    # Initialize construction terms
    kernel = None
    # Need to apply operations to kernel components
    operations = {'+':sklearn.gaussian_process.kernels.Sum,
                  '*':sklearn.gaussian_process.kernels.Product,
                  '^':sklearn.gaussian_process.kernels.Exponentiation}
    runningKernel = ""
    paranthetical = False # Considering kernel expressions with parantheses
    tempKernel = None
    operator = None
    leftIndex = None
    embedded = 0          # Tracks number of embedded parantheticals
    for index, char in enumerate(kernelStatement):
      if char == '(':
        paranthetical = True
        # Where does parantheses start?
        if leftIndex is None:
          leftIndex = index+1
        else:
          embedded += 1
          continue
      elif char ==')':
        # Want to handle parantheses at the highest level first
        if embedded > 0:
          embedded -= 1
          continue
        # Slicing out expression in parantheses
        parantheticalKernel = kernelStatement[leftIndex:index+embedded]
        tempKernel = self.customKernel(parantheticalKernel, lengthScaleSetting)
        paranthetical = False # ending paranthetical statement
        leftIndex = None
        continue
      # Just finish reading through statement in parantheses
      if paranthetical:
        continue
      # Is the character a piece of a kernel tag or an operator tag?
      if char not in ['+', '*', '^']:
        runningKernel += char
      else:
        # Consider exponentiation first
        if operator == '^':
          tempKernel = float(runningKernel)
        # Are we about to operate on a paranthetical statement?
        if tempKernel is None:
          tempKernel = self.pickKernel(runningKernel, lengthScaleSetting)

        if kernel is None:
          kernel = tempKernel
        else:
          kernel = operations[operator](kernel, tempKernel)
        # Resetting construction pieces and setting operator for next piece
        operator = char
        tempKernel = None
        runningKernel = ""
    # Need to apply final operation
    if len(runningKernel) > 0:
      try:
        if operator == '^':
           kernel = operations[operator](kernel, float(runningKernel))
        else:
          kernel = operations[operator](kernel, self.pickKernel(runningKernel, lengthScaleSetting))
      except KeyError:
        kernel = self.pickKernel(runningKernel, lengthScaleSetting)
    # Necessary if last term is a paranthetical
    else:
      try:
        kernel = operations[operator](kernel, tempKernel)
      except KeyError:
        kernel = tempKernel
    return kernel

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['kernel', 'alpha', 'n_restarts_optimizer', 'multioutput',
                                                               'normalize_y', 'random_state', 'optimizer', 'anisotropic', 'custom_kernel'])
    # notFound must be empty
    assert(not notFound)
    # special treatment for kernel
    if settings['kernel'] != "Custom":
      settings['kernel'] = self.pickKernel(settings['kernel'], settings['anisotropic']) if settings['kernel'] is not None else None
    # further special treatment for custom kernels
    else:
      if settings['custom_kernel'] == None:
        self.raiseAnError(OSError, 'Custom kernel selected but no custom_kernel input was provided.')
      settings['kernel'] = self.customKernel(settings['custom_kernel'], settings['anisotropic'])
    # Is this regressor for multi-output purposes?
    self.multioutputWrapper = settings['multioutput']
    # Deleting items that scikit-learn does not use
    del settings['multioutput']
    del settings['anisotropic']
    del settings['custom_kernel']
    self.initializeModel(settings)
