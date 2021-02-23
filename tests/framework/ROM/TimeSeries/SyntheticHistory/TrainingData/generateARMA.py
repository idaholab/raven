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
  Generates training data in the form of Fourier signals.
"""
import numpy as np
from generators import arma, toFile

plot = True

##########
# ARMA A #
##########
# normally-distributed noise with 0 loc, 1 scale

seconds = np.arange(1000) / 10. # 0 to 100 in 0.1 increments

slags = [0.4, 0.2]
nlags = [0.3, 0.2, 0.1]
signal0, _ = arma(slags, nlags, seconds, plot=plot)
slags = [0.5, 0.3]
nlags = [0.1, 0.05, 0.01]
signal1, _ = arma(slags, nlags, seconds, plot=plot)

if plot:
  import matplotlib.pyplot as plt
  plt.show()

out = np.zeros((len(seconds), 3))
out[:, 0] = seconds
out[:, 1] = signal0
out[:, 2] = signal1
toFile(out, 'ARMA_A', pivotName='seconds')
