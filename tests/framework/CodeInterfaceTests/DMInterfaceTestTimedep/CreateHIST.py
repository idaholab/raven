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
# -*- coding: utf-8 -*-
"""
created 11/16/16
$
@author: A. S. Epiney
"""

# This file assembles a couple of points into a vector

import numpy as np


def initialize(self, runInfoDict, inputFiles):

  pass


def run(self, Inputs):

  print "=============== Inside CreateHIST ================="
  self.G_vect = np.array([Inputs['G_a'][0], Inputs['G_b'][0], Inputs['G_c'][0]])
  print "=============== End CreateHIST ================="
