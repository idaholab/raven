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
  The Metrics module includes the different type of metrics
  to measure distance among RAVEN dataobjects
"""

## These lines ensure that we do not have to do something like:
## 'from OutStreamManagers.OutStreamPlot import OutStreamPlot' outside
## of this submodule
from .Metric import Metric
from .DTW import DTW
from .SklMetric import SKL
from .PairwiseMetric import PairwiseMetric
from .CDFAreaDifference import CDFAreaDifference
from .PDFCommonArea import PDFCommonArea
#from .RACDistance import RACDistance
from .ScipyMetric import ScipyMetric

from .Factory import factory
