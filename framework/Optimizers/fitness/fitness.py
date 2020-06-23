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
  Implementation of fitness function for Genetic Algorithm
  currently the implemented fitness function is a linear combination of the objective function and prenalty function for constraint violation:

  Created June,16,2020
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi
"""

def invLinear(rlz,**kwargs):
  """
    .. math::

    fitness = \\frac{1}{a * obj + b * penalty}

    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          obj, float, the value of the objectiove function at the chromosome for which fitness is computed
          a, float, linear coefficient for the objective function (default = 1.0)
          penalty, float, measuring the saverity of the constraint violation. (defazult = 1.0)
          b, float, linear coefficient for the penalty measure. (default = 1.0)
    @ Out, fitness, float, the fitness function of the given objective corresponding to a specific chromosome.
  """
  if kwargs['a'] == None:
    a = 1.0
  else:
    a = kwargs['a']
  if kwargs['b'] == None:
    b = 1.0
  else:
    b = kwargs['b']
  if kwargs['penalty'] == None:
    penalty = 0.0
  else:
    penalty = kwargs['penalty']

  objVar = kwargs['objVar']
  obj = eval(str('rlz.'+objVar+'.data'))
  fitness = 1/(a * obj + b * penalty)
  return fitness

__fitness = {}
__fitness['invLinear'] = invLinear


def returnInstance(cls, name):
  if name not in __fitness:
    cls.raiseAnError (IOError, "{} FITNESS FUNCTION NOT IMPLEMENTED!!!!!".format(name))
  return __fitness[name]
