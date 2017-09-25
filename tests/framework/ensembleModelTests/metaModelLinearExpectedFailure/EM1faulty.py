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
import numpy as np
from scipy.integrate import odeint
import copy
import time

###################################################################################################################
#                                                                                                                 #
#  THIS MODEL WILL ALWAYS FAIL (IT IS USED TO CHECK THE COLLECTION OF FAILED RUNS FROM THE ENSEMBLE MODEL)        #
#                                                                                                                 #
###################################################################################################################

def run(self, Input):
  raise RuntimeError("EXPECTED FAILURE")

