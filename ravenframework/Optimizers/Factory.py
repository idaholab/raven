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
  Created on May 21, 2016
  @author: chenj
"""

from ..EntityFactoryBase import EntityFactory

################################################################################
from .Optimizer import Optimizer
from .RavenSampled import RavenSampled
from .GradientDescent import GradientDescent
from .SimulatedAnnealing import SimulatedAnnealing
from .GeneticAlgorithm import GeneticAlgorithm
from .MultiObjectiveGeneticAlgorithm import MultiObjectiveGeneticAlgorithm
from .NSGAII import NSGAII


class OptimizerFactory(EntityFactory):
  """
    Optimizer entity factory. Extends the base factory so that the
    MultiObjectiveGeneticAlgorithm node selects its concrete algorithm
    (e.g. NSGA-II) through the 'type' attribute, while every concrete
    algorithm remains a subclass of MultiObjectiveGeneticAlgorithm.
  """
  def instanceFromXML(self, xml):
    """
      Using the provided XML, return the required instance. For the
      MultiObjectiveGeneticAlgorithm node the concrete algorithm subclass is
      resolved from the 'type' attribute (defaulting to 'NSGA-II'); all other
      optimizer nodes resolve by tag exactly as in the base factory.
      @ In, xml, xml.etree.ElementTree.Element, head element for instance
      @ Out, kind, str, name of type of entity (the node tag)
      @ Out, name, str, identifying name of entity
      @ Out, entity, instance, object from factory
    """
    kind = xml.tag
    name = xml.attrib['name']
    if kind == 'MultiObjectiveGeneticAlgorithm':
      algorithmType = xml.attrib.get('type', 'NSGA-II')
      algorithmClass = MultiObjectiveGeneticAlgorithm.knownAlgorithms.get(algorithmType)
      if algorithmClass is None:
        self.raiseAnError(IOError, f'<MultiObjectiveGeneticAlgorithm> has unknown type "{algorithmType}"; '
                          f'known types are: {", ".join(MultiObjectiveGeneticAlgorithm.knownAlgorithms)}')
      entity = algorithmClass()
    else:
      entity = self.returnInstance(kind)
    return kind, name, entity


factory = OptimizerFactory('Optimizer')
factory.registerType('GradientDescent', GradientDescent)
factory.registerType('SimulatedAnnealing', SimulatedAnnealing)
factory.registerType('GeneticAlgorithm', GeneticAlgorithm)
factory.registerType('MultiObjectiveGeneticAlgorithm', MultiObjectiveGeneticAlgorithm)
# Concrete multi-objective GA algorithms are registered against the MultiObjectiveGeneticAlgorithm
# base class and selected at input time via <MultiObjectiveGeneticAlgorithm type="...">.
# Adding a new variant (e.g. NSGA-III) is a single registration line here plus its subclass.
MultiObjectiveGeneticAlgorithm.registerAlgorithm('NSGA-II', NSGAII)

try:
    from .BayesianOptimizer import BayesianOptimizer
    factory.registerType('BayesianOptimizer', BayesianOptimizer)
except ModuleNotFoundError as error:
    print("ERROR: Unable to import BayesianOptimizer", error)
