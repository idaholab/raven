# BSD 3-Clause License

# Copyright (c) 2016-2020 The scikit-optimize developers.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
  Created on June 22, 2023

  @author: grifaa
  Kernels for Gaussian processes (GP's)
"""

# FIXME How to give credit to scikit-optimize for the code

#External Modules
from math import sqrt
import numpy as np
from sklearn.gaussian_process.kernels import Kernel as sk_Kernel
from sklearn.gaussian_process.kernels import ConstantKernel as sk_ConstantKernel
from sklearn.gaussian_process.kernels import DotProduct as sk_DotProduct
from sklearn.gaussian_process.kernels import Exponentiation as sk_Exponentiation
from sklearn.gaussian_process.kernels import ExpSineSquared as sk_ExpSineSquared
from sklearn.gaussian_process.kernels import Matern as sk_Matern
from sklearn.gaussian_process.kernels import Product as sk_Product
from sklearn.gaussian_process.kernels import RationalQuadratic as sk_RationalQuadratic
from sklearn.gaussian_process.kernels import RBF as sk_RBF
from sklearn.gaussian_process.kernels import Sum as sk_Sum
from sklearn.gaussian_process.kernels import WhiteKernel as sk_WhiteKernel
#External Modules

class Kernel(sk_Kernel):
  """
    Base class for skopt.gaussian_process kernels.
    Supports computation of the gradient of the kernel with respect to X
  """
  def __add__(self, b):
    """
      Overloads addition operator for kernel objects, to apply correct operation
      @ In, b, typically a kernel object, element to apply addition to
      @ Out, resultant kernel object from applying addition
    """
    if not isinstance(b, Kernel):
      return Sum(self, ConstantKernel(b))
    return Sum(self, b)

  def __radd__(self, b):
    """
      Overloads series addition operator for kernel objects, to apply correct operation
      @ In, b, typically a kernel object, element to apply series addition to
      @ Out, resultant kernel object from applying series addition
    """
    if not isinstance(b, Kernel):
      return Sum(ConstantKernel(b), self)
    return Sum(b, self)

  def __mul__(self, b):
    """
      Overloads multiplication operator for kernel objects, to apply correct operation
      @ In, b, typically a kernel object, element to apply multiplication to
      @ Out, resultant kernel object from applying multiplication
    """
    if not isinstance(b, Kernel):
      return Product(self, ConstantKernel(b))
    return Product(self, b)

  def __rmul__(self, b):
    """
      Overloads series multiplication operator for kernel objects, to apply correct operation
      @ In, b, typically a kernel object, element to apply series multiplication to
      @ Out, resultant kernel object from applying series multiplication
    """
    if not isinstance(b, Kernel):
      return Product(ConstantKernel(b), self)
    return Product(b, self)

  def __pow__(self, b):
    """
      Overloads exponentiation operator for kernel objects, to apply correct operation
      @ In, b, typically a kernel object, element to apply exponentiation to
      @ Out, resultant kernel object from applying exponentiation
    """
    return Exponentiation(self, b)

  def gradient_x(self, x, X_train):
    """
      Computes gradient of K(x, X_train) with respect to x
      @ In, x, array-like, shape=(n_features,), test point
      @ In, X_train, array-like, shape=(n_samples, n_features), Training data used in fitting GPR
      @ Out, gradient_x, array-like, shape=(n_samples, n_features), Gradient of K(x, X_train) wrt x
    """
    # NOTE for kernels that may not have implemented this method
    raise NotImplementedError

class RBF(Kernel, sk_RBF):
  """
    Extends RBF kernel to include gradients wrt to input space (x)
  """
  def gradient_x(self, x, X_train):
    """
      Computes gradient of K(x, X_train) with respect to x
      @ In, x, array-like, shape=(n_features,), test point
      @ In, X_train, array-like, shape=(n_samples, n_features), Training data used in fitting GPR
      @ Out, gradient_x, array-like, shape=(n_samples, n_features), Gradient of K(x, X_train) wrt x
    """
    # diff = (x - X) / length_scale
    # size = (n_train_samples, n_dimensions)
    x = np.asarray(x)
    X_train = np.asarray(X_train)

    length_scale = np.asarray(self.length_scale)
    diff = x - X_train
    diff /= length_scale

    # e = -exp(0.5 * \sum_{i=1}^d (diff ** 2))
    # size = (n_train_samples, 1)
    exp_diff_squared = np.sum(diff**2, axis=1)
    exp_diff_squared *= -0.5
    exp_diff_squared = np.exp(exp_diff_squared, exp_diff_squared)
    exp_diff_squared = np.expand_dims(exp_diff_squared, axis=1)
    exp_diff_squared *= -1

    # gradient = (e * diff) / length_scale
    gradient = exp_diff_squared * diff
    gradient /= length_scale
    return gradient

class Matern(Kernel, sk_Matern):
  """
    Extends Matern kernel to include gradients wrt to input space (x)
  """
  def gradient_x(self, x, X_train):
    """
      Computes gradient of K(x, X_train) with respect to x
      @ In, x, array-like, shape=(n_features,), test point
      @ In, X_train, array-like, shape=(n_samples, n_features), Training data used in fitting GPR
      @ Out, gradient_x, array-like, shape=(n_samples, n_features), Gradient of K(x, X_train) wrt x
    """
    x = np.asarray(x)
    X_train = np.asarray(X_train)
    length_scale = np.asarray(self.length_scale)

    # diff = (x - X_train) / length_scale
    # size = (n_train_samples, n_dimensions)
    diff = x - X_train
    diff /= length_scale

    # dist_sq = \sum_{i=1}^d (diff ^ 2)
    # dist = sqrt(dist_sq)
    # size = (n_train_samples,)
    dist_sq = np.sum(diff**2, axis=1)
    dist = np.sqrt(dist_sq)

    if self.nu == 0.5:
      # e = -np.exp(-dist) / dist
      # size = (n_train_samples, 1)
      scaled_exp_dist = -dist
      scaled_exp_dist = np.exp(scaled_exp_dist, scaled_exp_dist)
      scaled_exp_dist *= -1

      # grad = (e * diff) / length_scale
      # For all i in [0, D) if x_i equals y_i.
      # 1. e -> -1
      # 2. (x_i - y_i) / \sum_{j=1}^D (x_i - y_i)**2 approaches 1.
      # Hence the gradient when for all i in [0, D),
      # x_i equals y_i is -1 / length_scale[i].
      gradient = -np.ones((X_train.shape[0], x.shape[0]))
      mask = dist != 0.0
      scaled_exp_dist[mask] /= dist[mask]
      scaled_exp_dist = np.expand_dims(scaled_exp_dist, axis=1)
      gradient[mask] = scaled_exp_dist[mask] * diff[mask]
      gradient /= length_scale
      return gradient

    elif self.nu == 1.5:
      # grad(fg) = f'g + fg'
      # where f = 1 + sqrt(3) * euclidean((X - Y) / length_scale)
      # where g = exp(-sqrt(3) * euclidean((X - Y) / length_scale))
      sqrt_3_dist = sqrt(3) * dist
      f = np.expand_dims(1 + sqrt_3_dist, axis=1)

      # When all of x_i equals y_i, f equals 1.0, (1 - f) equals
      # zero, hence from below
      # f * g_grad + g * f_grad (where g_grad = -g * f_grad)
      # -f * g * f_grad + g * f_grad
      # g * f_grad * (1 - f) equals zero.
      # sqrt_3_by_dist can be set to any value since diff equals
      # zero for this corner case.
      sqrt_3_by_dist = np.zeros_like(dist)
      nzd = dist != 0.0
      sqrt_3_by_dist[nzd] = sqrt(3) / dist[nzd]
      dist_expand = np.expand_dims(sqrt_3_by_dist, axis=1)

      f_grad = diff / length_scale
      f_grad *= dist_expand

      sqrt_3_dist *= -1
      exp_sqrt_3_dist = np.exp(sqrt_3_dist, sqrt_3_dist)
      g = np.expand_dims(exp_sqrt_3_dist, axis=1)
      g_grad = -g * f_grad

      # f * g_grad + g * f_grad (where g_grad = -g * f_grad)
      f *= -1
      f += 1
      return g * f_grad * f

    elif self.nu == 2.5:
      # grad(fg) = f'g + fg'
      # where f = (1 + sqrt(5) * euclidean((X - Y) / length_scale) +
      #            5 / 3 * sqeuclidean((X - Y) / length_scale))
      # where g = exp(-sqrt(5) * euclidean((X - Y) / length_scale))
      sqrt_5_dist = sqrt(5) * dist
      f2 = (5.0 / 3.0) * dist_sq
      f2 += sqrt_5_dist
      f2 += 1
      f = np.expand_dims(f2, axis=1)

      # For i in [0, D) if x_i equals y_i
      # f = 1 and g = 1
      # Grad = f'g + fg' = f' + g'
      # f' = f_1' + f_2'
      # Also g' = -g * f1'
      # Grad = f'g - g * f1' * f
      # Grad = g * (f' - f1' * f)
      # Grad = f' - f1'
      # Grad = f2' which equals zero when x = y
      # Since for this corner case, diff equals zero,
      # dist can be set to anything.
      nzd_mask = dist != 0.0
      nzd = dist[nzd_mask]
      dist[nzd_mask] = np.reciprocal(nzd, nzd)

      dist *= sqrt(5)
      dist = np.expand_dims(dist, axis=1)
      diff /= length_scale
      f1_grad = dist * diff
      f2_grad = (10.0 / 3.0) * diff
      f_grad = f1_grad + f2_grad

      sqrt_5_dist *= -1
      g = np.exp(sqrt_5_dist, sqrt_5_dist)
      g = np.expand_dims(g, axis=1)
      g_grad = -g * f1_grad
      return f * g_grad + g * f_grad

class RationalQuadratic(Kernel, sk_RationalQuadratic):
  """
    Extends RationalQuadratic kernel to include gradients wrt to input space (x)
  """
  def gradient_x(self, x, X_train):
    """
      Computes gradient of K(x, X_train) with respect to x
      @ In, x, array-like, shape=(n_features,), test point
      @ In, X_train, array-like, shape=(n_samples, n_features), Training data used in fitting GPR
      @ Out, gradient_x, array-like, shape=(n_samples, n_features), Gradient of K(x, X_train) wrt x
    """
    x = np.asarray(x)
    X_train = np.asarray(X_train)
    alpha = self.alpha
    length_scale = self.length_scale

    # diff = (x - X_train) / length_scale
    # size = (n_train_samples, n_dimensions)
    diff = x - X_train
    diff /= length_scale

    # dist = -(1 + (\sum_{i=1}^d (diff^2) / (2 * alpha)))** (-alpha - 1)
    # size = (n_train_samples,)
    scaled_dist = np.sum(diff**2, axis=1)
    scaled_dist /= (2 * self.alpha)
    scaled_dist += 1
    scaled_dist **= (-alpha - 1)
    scaled_dist *= -1

    scaled_dist = np.expand_dims(scaled_dist, axis=1)
    diff_by_ls = diff / length_scale
    return scaled_dist * diff_by_ls

class ExpSineSquared(Kernel, sk_ExpSineSquared):
  """
    Extends ExpSineSquared kernel to include gradients wrt to input space (x)
  """
  def gradient_x(self, x, X_train):
    """
      Computes gradient of K(x, X_train) with respect to x
      @ In, x, array-like, shape=(n_features,), test point
      @ In, X_train, array-like, shape=(n_samples, n_features), Training data used in fitting GPR
      @ Out, gradient_x, array-like, shape=(n_samples, n_features), Gradient of K(x, X_train) wrt x
    """
    x = np.asarray(x)
    X_train = np.asarray(X_train)
    length_scale = self.length_scale
    periodicity = self.periodicity

    diff = x - X_train
    sq_dist = np.sum(diff**2, axis=1)
    dist = np.sqrt(sq_dist)

    pi_by_period = dist * (np.pi / periodicity)
    sine = np.sin(pi_by_period) / length_scale
    sine_squared = -2 * sine**2
    exp_sine_squared = np.exp(sine_squared)

    grad_wrt_exp = -2 * np.sin(2 * pi_by_period) / length_scale**2

    # When x_i -> y_i for all i in [0, D), the gradient becomes
    # zero. See https://github.com/MechCoder/Notebooks/blob/master/ExpSineSquared%20Kernel%20gradient%20computation.ipynb
    # for a detailed math explanation
    # grad_wrt_theta can be anything since diff is zero
    # for this corner case, hence we set to zero.
    grad_wrt_theta = np.zeros_like(dist)
    nzd = dist != 0.0
    grad_wrt_theta[nzd] = np.pi / (periodicity * dist[nzd])
    return np.expand_dims(
        grad_wrt_theta * exp_sine_squared * grad_wrt_exp, axis=1) * diff

class ConstantKernel(Kernel, sk_ConstantKernel):
  """
    Extends ConstantKernel kernel to include gradients wrt to input space (x)
  """
  def gradient_x(self, x, X_train):
    """
      Computes gradient of K(x, X_train) with respect to x
      @ In, x, array-like, shape=(n_features,), test point
      @ In, X_train, array-like, shape=(n_samples, n_features), Training data used in fitting GPR
      @ Out, gradient_x, array-like, shape=(n_samples, n_features), Gradient of K(x, X_train) wrt x
    """
    return np.zeros_like(X_train)

class WhiteKernel(Kernel, sk_WhiteKernel):
  """
    Extends WhiteKernel kernel to include gradients wrt to input space (x)
  """
  def gradient_x(self, x, X_train):
    """
      Computes gradient of K(x, X_train) with respect to x
      @ In, x, array-like, shape=(n_features,), test point
      @ In, X_train, array-like, shape=(n_samples, n_features), Training data used in fitting GPR
      @ Out, gradient_x, array-like, shape=(n_samples, n_features), Gradient of K(x, X_train) wrt x
    """
    return np.zeros_like(X_train)

class Exponentiation(Kernel, sk_Exponentiation):
  """
    Extends Exponentiation operation kernel to include gradients wrt to input space (x)
  """
  def gradient_x(self, x, X_train):
    """
      Computes gradient of K(x, X_train) with respect to x
      @ In, x, array-like, shape=(n_features,), test point
      @ In, X_train, array-like, shape=(n_samples, n_features), Training data used in fitting GPR
      @ Out, gradient_x, array-like, shape=(n_samples, n_features), Gradient of K(x, X_train) wrt x
    """
    x = np.asarray(x)
    X_train = np.asarray(X_train)
    expo = self.exponent
    kernel = self.kernel

    K = np.expand_dims(
        kernel(np.expand_dims(x, axis=0), X_train)[0], axis=1)
    return expo * K ** (expo - 1) * kernel.gradient_x(x, X_train)

class Sum(Kernel, sk_Sum):
  """
    Extends Sum operation kernel to include gradients wrt to input space (x)
  """
  def gradient_x(self, x, X_train):
    """
      Computes gradient of K(x, X_train) with respect to x
      @ In, x, array-like, shape=(n_features,), test point
      @ In, X_train, array-like, shape=(n_samples, n_features), Training data used in fitting GPR
      @ Out, gradient_x, array-like, shape=(n_samples, n_features), Gradient of K(x, X_train) wrt x
    """
    return (
        self.k1.gradient_x(x, X_train) +
        self.k2.gradient_x(x, X_train)
    )

class Product(Kernel, sk_Product):
  """
    Extends Product operation kernel to include gradients wrt to input space (x)
  """
  def gradient_x(self, x, X_train):
    """
      Computes gradient of K(x, X_train) with respect to x
      @ In, x, array-like, shape=(n_features,), test point
      @ In, X_train, array-like, shape=(n_samples, n_features), Training data used in fitting GPR
      @ Out, gradient_x, array-like, shape=(n_samples, n_features), Gradient of K(x, X_train) wrt x
    """
    x = np.asarray(x)
    x = np.expand_dims(x, axis=0)
    X_train = np.asarray(X_train)
    f_ggrad = (
        np.expand_dims(self.k1(x, X_train)[0], axis=1) *
        self.k2.gradient_x(x, X_train)
    )
    fgrad_g = (
        np.expand_dims(self.k2(x, X_train)[0], axis=1) *
        self.k1.gradient_x(x, X_train)
    )
    return f_ggrad + fgrad_g

class DotProduct(Kernel, sk_DotProduct):
  """
    Extends DotProduct kernel to include gradients wrt to input space (x)
  """
  def gradient_x(self, x, X_train):
    """
      Computes gradient of K(x, X_train) with respect to x
      @ In, x, array-like, shape=(n_features,), test point
      @ In, X_train, array-like, shape=(n_samples, n_features), Training data used in fitting GPR
      @ Out, gradient_x, array-like, shape=(n_samples, n_features), Gradient of K(x, X_train) wrt x
    """
    return np.asarray(X_train)
