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
  The SupervisedLearning module includes different types of ROM strategies available in RAVEN
"""

from .SupervisedLearning import SupervisedLearning
from .ScikitLearn.ScikitLearnBase import ScikitLearnBase
from .KerasBase import KerasBase
from .KerasRegression import KerasRegression
from .KerasClassifier import KerasClassifier
from .ROMCollection import Collection
from .NDinterpolatorRom  import NDinterpolatorRom

from .Factory import factory
