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
  Converts multi-objective <GeneticAlgorithm> optimizers to <MultiObjectiveGeneticAlgorithm type="NSGA-II">.

  Previously a <GeneticAlgorithm> whose <objective> listed more than one variable ran NSGA-II.
  <GeneticAlgorithm> is now single-objective, and multi-objective runs use
  <MultiObjectiveGeneticAlgorithm type="NSGA-II">. This script renames every multi-objective
  <GeneticAlgorithm> under <Optimizers> and updates each <Steps> reference to it
  (<Optimizer class="Optimizers" type="GeneticAlgorithm">name</Optimizer>). Single-objective
  <GeneticAlgorithm> nodes are left unchanged.

  Usage:
    python path/to/toMultiObjectiveGA.py input1.xml input2.xml ...
"""
import xml.etree.ElementTree as ET

def _isMultiObjective(optNode):
  """
    Whether an optimizer node lists more than one objective variable.
    @ In, optNode, xml.etree.ElementTree.Element, an optimizer node
    @ Out, isMulti, bool, True if <objective> names two or more variables
  """
  objective = optNode.find('objective')
  if objective is None or objective.text is None:
    return False
  return len([obj for obj in objective.text.split(',') if obj.strip()]) > 1

def convert(tree, fileName=None):
  """
    Converts multi-objective <GeneticAlgorithm> nodes to <MultiObjectiveGeneticAlgorithm type="NSGA-II">.
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, str, optional, the name of the RAVEN input file
    @ Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  if simulation.tag != 'Simulation':
    return tree # not a RAVEN input file
  optimizers = simulation.find('Optimizers')
  if optimizers is None:
    return tree
  converted = set()
  for opt in optimizers.findall('GeneticAlgorithm'):
    if _isMultiObjective(opt):
      opt.tag = 'MultiObjectiveGeneticAlgorithm'
      opt.attrib.setdefault('type', 'NSGA-II')
      converted.add(opt.attrib.get('name'))
  if not converted:
    return tree
  steps = simulation.find('Steps')
  if steps is not None:
    for step in steps:
      for ref in step.findall('Optimizer'):
        if ref.attrib.get('type') == 'GeneticAlgorithm' and (ref.text or '').strip() in converted:
          ref.attrib['type'] = 'MultiObjectiveGeneticAlgorithm'
  return tree

if __name__ == '__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv, convert)
