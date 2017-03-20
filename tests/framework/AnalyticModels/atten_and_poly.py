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
#***************************************
#* Simple analytic test ExternalModule *
#***************************************
#
#  Combines both tensor_poly and attenuate for simple evaluation
#
import numpy as np
import tensor_poly
import attenuate

def evaluate(inp):
  return tensor_poly.evaluate(inp)

def evaluate2(inp):
  return attenuate.evaluate(inp)

def run(self,Input):
  self.ans  = evaluate (Input.values())
  self.ans2 = evaluate2(Input.values())

#
#  These tests have analytic mean and variance, documented in raven/doc/tests
#
