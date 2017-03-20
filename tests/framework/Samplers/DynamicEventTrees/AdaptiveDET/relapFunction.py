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
'''
Created on Oct 20, 2013

@author: alfoa
'''
import numpy as np
import math

def __residuum__(self):
  print('variables in function '+str(self.__varType__))
  return

def __gradient__(self):
  print('variables in function '+str(self.__varType__))
  return

def __supportBoundingTest__(self):
  print('variables in function '+str(self.__varType__))
  return

def __residuumSign(self):
  print('CLAD DAMAGED IS ' + str(self.CladDamaged))
  if self.CladDamaged == 1: return -1.0
  else: return 1.0
