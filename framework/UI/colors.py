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
    This provides some convenient colors to be used in different UIs including
    a color-blind safe color map and consistent and aesthetically pleasing cool
    and warm (blue and red) colors.
"""

#For future compatibility with Python 3
from __future__ import division, print_function, absolute_import
#End compatibility block for Python 3

from matplotlib import cm, colors
import numpy as np
import itertools

try:
  from PySide.QtGui import QColor
except ImportError as e:
  from PySide2.QtGui import QColor

minPenColor = QColor(33,102,172)
minBrushColor = QColor(67,147,195)
inactiveMinPenColor = minBrushColor.lighter()
inactiveMinBrushColor = minBrushColor.lighter()

maxPenColor = QColor(178,24,43)
maxBrushColor = QColor(214,96,77)
inactiveMaxPenColor = maxBrushColor.lighter()
inactiveMaxBrushColor = maxBrushColor.lighter()


TolColors = ['#88CCEE', '#DDCC77', '#AA4499', '#117733', '#332288', '#999933',
             '#44AA99', '#882255', '#CC6677']

colorCycle = itertools.cycle(TolColors)
