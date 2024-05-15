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
import imp
import os
# the functions below are imported from the respective files
# this is aimed to show how multiple functions can be coded in the same file
d_calc = imp.load_source('d_calc', os.path.join(os.path.dirname(os.path.abspath(__file__)),'../RedundantInputs/d_calc.py'))

def raven_d_calc_f_a_c(ravenContainer):
  return d_calc.evaluate(ravenContainer)

def raven_e_calc_f_a_c(ravenContainer):
  return ravenContainer.a + ravenContainer.c

def raven_b_calc_f_a(ravenContainer):
  return ravenContainer.a*1.2

def raven_c_calc_f_b(ravenContainer):
  return ravenContainer.b*1.2

def raven_z_l_calc_f_a(ravenContainer):
  return ravenContainer.a*1.2

