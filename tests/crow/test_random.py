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

engine = distribution1D.DistributionContainer.instance(True)

results = {"pass":0,"fail":0}

# test seeds, consistency check
# seed 42, first two entries are 0.37450114397,  0.796542984386
# seed 65, first two entries are 0.218645046982, 0.869204071786
engine.seedRandom(42)
utils.checkAnswer('First seed, first number',engine.random(),0.374540114397,results)
engine.seedRandom(65)
utils.checkAnswer('second seed, first number',engine.random(),0.218645046982,results)

# test sampled values
## set a seed
seed = 42
engine.seedRandom(seed)
## get a few values
rands = list(engine.random() for _ in range(5))
expected = [0.37454011439684315, 0.7965429843861012, 0.9507143117838339, 0.1834347877147223, 0.7319939385009916]
for i in range(5):
  utils.checkAnswer("random {}".format(i),rands[i],expected[i],results)
# test shifting values
## move random number count forward (starting at same seed)
shift = 2 # amount to shift evaluations by
engine.seedRandom(seed,shift)
rands2 = list(engine.random() for _ in range(5))
for i in range(5-shift):
  utils.checkAnswer("progressed seed {}".format(i),rands[i+shift],rands2[i],results)

# test two independent engines
## make engines, set seeds
eng1 = distribution1D.DistributionContainer.instance(True)
eng2 = distribution1D.DistributionContainer.instance(True)
seed = 314159
eng1.seedRandom(seed)
eng2.seedRandom(seed)
# first 6 seeds for 314159 are: 0.81792331017, 0.0776717064617, 0.551046290098, 0.997661588247, 0.419775358732, 0.385839152938
## check first three samples are the same for each
for i in range(3):
  utils.checkAnswer("Parallel engines {}".format(i),eng1.random(),eng2.random(),results)
## check different with different seeds
eng1.seedRandom(65535)
eng2.seedRandom(512)
rand1 = eng1.random()
rand2 = eng2.random()
utils.checkAnswer("Different seeds 1",rand1,0.193341771651,results)
utils.checkAnswer("Different seeds 2",rand2,0.107295856603,results)

print(results)

sys.exit(results["fail"])
