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
#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import sys
import utils

distribution1D = utils.find_distribution1D()

engine = distribution1D.DistributionContainer.instance()

results = {"pass":0,"fail":0}


# set a seed
seed = 42
engine.seedRandom(seed)
# get a few values
rands = list(engine.random() for _ in range(5))
expected = [0.37454011439684315, 0.7965429843861012, 0.9507143117838339, 0.1834347877147223, 0.7319939385009916]
for i in range(5):
  utils.checkAnswer("random {}".format(i),rands[i],expected[i],results)
# move random number count forward (starting at same seed)
shift = 2 # amount to shift evaluations by
engine.seedRandom(seed,shift)
rands2 = list(engine.random() for _ in range(5))
for i in range(5-shift):
  utils.checkAnswer("progressed seed {}".format(i),rands[i+shift],rands2[i],results)


print(results)

sys.exit(results["fail"])
