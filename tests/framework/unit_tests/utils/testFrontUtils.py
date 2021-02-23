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
  This Module performs Unit Tests for the frontUtils methods
"""


import os,sys
import numpy as np
frameworkDir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),os.pardir,os.pardir,os.pardir,os.pardir,'framework'))
sys.path.append(frameworkDir)
from utils import utils
utils.find_crow(frameworkDir)
from utils import frontUtils
import testMathUtils as tMU

randomENG = utils.findCrowModule("randomENG")

print (frontUtils)

results = {"pass":0,"fail":0}


test = np.array([ 0.21573114, -0.92937786,  0.29952775],
                [ 0.94716548, -0.31085637, -0.07903087],
                [ 0.6485263,  -0.72106429,  0.24388507],
                [ 0.3466882,  -0.78716832,  0.51007189],
                [ 0.15463182, -0.18730138,  0.97005525],
                [ 0.02937279, -0.91175393,  0.40968525],
                [-0.24039731,  0.54889384,  0.80057772],
                [ 0.06213356,  0.28552822, -0.95635404],
                [-0.20190017,  0.66695686, -0.71722024],
                [-0.62399932, -0.22858416,  0.74724436])

mask = frontUtils.nonDominatedFrontier(test, returnMask=True)
answerMask = np.array([True, True, True, False, False, True, False, True, True, True])
tMU.checkArray('nonDominatedFrontier with Mask', mask, answerMask)

indexes = frontUtils.nonDominatedFrontier(test, returnMask=False)
answerIndexes = np.array([0, 1, 2, 5, 7, 8, 9])
tMU.checkArray('nonDominatedFrontier with indexes', indexes, answerIndexes)

sys.exit(results["fail"])

