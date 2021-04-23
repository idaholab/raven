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
  Created on May 8, 2018

  @author: talbpaul
  Originally from SupervisedLearning.py, split in PR #650 in July 2018
  Specific ROM implementation for NDinvDistWeight ROM
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils
interpolationND = utils.findCrowModule("interpolationND")
from .NDinterpolatorRom import NDinterpolatorRom
#Internal Modules End--------------------------------------------------------------------------------

class NDinvDistWeight(NDinterpolatorRom):
  """
    An N-dimensional model that interpolates data based on a inverse weighting of
    their training data points?
  """
  ROMtype         = 'NDinvDistWeight'
  def __init__(self, **kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    NDinterpolatorRom.__init__(self, **kwargs)
    self.printTag = 'ND-INVERSEWEIGHT ROM'
    if not 'p' in self.initOptionDict.keys():
      self.raiseAnError(IOError,'the <p> parameter must be provided in order to use NDinvDistWeigth as ROM!!!!')
    self.__initLocal__()

  def __initLocal__(self):
    """
      Method used to add additional initialization features used by pickling
      @ In, None
      @ Out, None
    """
    self.interpolator = []
    for _ in range(len(self.target)):
      self.interpolator.append(interpolationND.InverseDistanceWeighting(float(self.initOptionDict['p'])))

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    self.__initLocal__()
