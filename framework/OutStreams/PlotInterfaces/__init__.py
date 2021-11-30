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
  Implementations of plottings strategies for OutStreams.

  Created April 1, 2021
  @author: talbpaul
"""
from .PlotInterface import PlotInterface
from .SamplePlot import SamplePlot
from .GeneralPlot import GeneralPlot as Plot
from .OptPath import OptPath
from .SyntheticCloud import SyntheticCloud
from .PopulationPlot import PopulationPlot
from. OptParallelCoordinate import OptParallelCoordinatePlot

from .Factory import factory
