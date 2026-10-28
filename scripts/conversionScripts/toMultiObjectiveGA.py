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
  Converts genetic-algorithm inputs written for RAVEN before the GA refactor to the current syntax.

  Before the refactor, a <GeneticAlgorithm> whose <objective> listed more than one variable ran
  NSGA-II. <GeneticAlgorithm> is now single-objective, and multi-objective runs use
  <MultiObjectiveGeneticAlgorithm type="NSGA-II">. For every multi-objective <GeneticAlgorithm>
  under <Optimizers>, this script:
    - renames it to <MultiObjectiveGeneticAlgorithm type="NSGA-II">;
    - updates each <Steps> reference to it (<Optimizer class="Optimizers" type="GeneticAlgorithm">);
    - reports settings that NSGA-II rejects or ignores, so they can be reviewed by hand.
  Single-objective <GeneticAlgorithm> nodes are left unchanged, except that one without a <fitness>
  node is reported: a single-objective GA now needs one (fitness is its selection criterion).

  Only those tags are edited, in place, so comments, attribute quoting, indentation and blank
  lines are kept exactly. A file with nothing to convert is not rewritten.

  Usage:
    python path/to/toMultiObjectiveGA.py input1.xml input2.xml ...
    python path/to/toMultiObjectiveGA.py --check input1.xml ...   (report only, write nothing)
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

# NSGA-II settings that the multi-objective GA requires
_MULTI_PARENT_SELECTION = 'tournamentSelection'
_MULTI_SURVIVOR_SELECTION = 'rankNcrowdingBased'


def _objectives(optNode):
  """
    Objective variable names of an optimizer node.
    @ In, optNode, xml.etree.ElementTree.Element, an optimizer node
    @ Out, objectives, list(str), objective names
  """
  objective = optNode.find('objective')
  if objective is None or objective.text is None:
    return []
  return [obj.strip() for obj in objective.text.split(',') if obj.strip()]


def _isMultiObjective(optNode):
  """
    Whether an optimizer node lists more than one objective variable.
    @ In, optNode, xml.etree.ElementTree.Element, an optimizer node
    @ Out, isMulti, bool, True if <objective> names two or more variables
  """
  return len(_objectives(optNode)) > 1


def _text(node, path):
  """
    Stripped text of a sub-node, or None.
    @ In, node, xml.etree.ElementTree.Element, parent
    @ In, path, str, sub-node path
    @ Out, text, str or None, its text
  """
  sub = node.find(path)
  return sub.text.strip() if sub is not None and sub.text else None


def review(optNode):
  """
    Settings in a converted optimizer that NSGA-II rejects, or that change meaning.
    @ In, optNode, xml.etree.ElementTree.Element, a multi-objective <GeneticAlgorithm> node
    @ Out, notes, list(str), one note per setting to review by hand
  """
  notes = []
  name = optNode.attrib.get('name', '?')
  parent = _text(optNode, 'GAparams/parentSelection')
  if parent is not None and parent != _MULTI_PARENT_SELECTION:
    notes.append(f'{name}: <parentSelection>{parent}</parentSelection> is rejected by NSGA-II; '
                 f'use {_MULTI_PARENT_SELECTION}.')
  survivor = _text(optNode, 'GAparams/survivorSelection')
  if survivor is not None and survivor != _MULTI_SURVIVOR_SELECTION:
    notes.append(f'{name}: <survivorSelection>{survivor}</survivorSelection> is rejected by NSGA-II; '
                 f'use {_MULTI_SURVIVOR_SELECTION}.')
  if optNode.find('GAparams/fitness') is not None:
    notes.append(f'{name}: <fitness> is kept but no longer drives selection. NSGA-II ranks by constrained '
                 f'domination; the fitness value is only exported. To rank by a penalized fitness instead, add '
                 f'<rankingAlgorithm type="feasibleFirstPenalty"/>.')
  return notes


def _findSingleObjectiveWithoutFitness(simulation):
  """
    Single-objective <GeneticAlgorithm> nodes without the now-required <fitness>.
    @ In, simulation, xml.etree.ElementTree.Element, root
    @ Out, names, list(str), optimizer names
  """
  optimizers = simulation.find('Optimizers')
  if optimizers is None:
    return []
  return [opt.attrib.get('name', '?') for opt in optimizers.findall('GeneticAlgorithm')
          if not _isMultiObjective(opt) and opt.find('GAparams/fitness') is None]


def convert(tree, fileName=None):
  """
    Converts multi-objective <GeneticAlgorithm> nodes to <MultiObjectiveGeneticAlgorithm type="NSGA-II">
    in an element tree. Kept for the conversion-script interface (convert_utils.standardMain); the
    command line uses convertText, which preserves formatting.
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


def convertText(text):
  """
    Converts the text of a RAVEN input file, editing only the tags that change.
    @ In, text, str, file contents
    @ Out, newText, str, converted contents (== text when nothing changes)
    @ Out, converted, list(str), names of the converted optimizers
    @ Out, notes, list(str), settings to review by hand
  """
  simulation = ET.fromstring(text)
  if simulation.tag != 'Simulation':
    return text, [], []
  notes = [f'{name}: single-objective <GeneticAlgorithm> has no <fitness>; one is now required '
           f'(fitness is the selection criterion).' for name in _findSingleObjectiveWithoutFitness(simulation)]
  optimizers = simulation.find('Optimizers')
  if optimizers is None:
    return text, [], notes
  targets = [opt for opt in optimizers.findall('GeneticAlgorithm') if _isMultiObjective(opt)]
  if not targets:
    return text, [], notes
  names = [opt.attrib.get('name') for opt in targets]
  for opt in targets:
    notes.extend(review(opt))
  newText = text
  for name in names:
    quoted = re.escape(name)
    # opening tag carrying this name (either quote style), and its matching closing tag
    opening = re.compile(r'<GeneticAlgorithm(\s[^>]*?\bname\s*=\s*([\'"])' + quoted + r'\2[^>]*)>')
    match = opening.search(newText)
    if match is None:
      raise ValueError(f'could not locate <GeneticAlgorithm name="{name}"> in the text')
    attrs = match.group(1)
    if not re.search(r'\btype\s*=', attrs):
      attrs += ' type="NSGA-II"'
    start = match.start()
    newText = newText[:start] + f'<MultiObjectiveGeneticAlgorithm{attrs}>' + newText[match.end():]
    close = newText.find('</GeneticAlgorithm>', start)
    if close < 0:
      raise ValueError(f'could not locate </GeneticAlgorithm> for "{name}"')
    newText = newText[:close] + '</MultiObjectiveGeneticAlgorithm>' + newText[close + len('</GeneticAlgorithm>'):]
    # <Steps> references: <Optimizer ... type="GeneticAlgorithm" ...> name </Optimizer>
    reference = re.compile(r'(<Optimizer\b[^>]*?\btype\s*=\s*([\'"]))GeneticAlgorithm(\2[^>]*>\s*' + quoted + r'\s*</Optimizer>)')
    newText = reference.sub(r'\1MultiObjectiveGeneticAlgorithm\3', newText)
  # the result must still be a valid deck with the same structure
  check = ET.fromstring(newText)
  assert sum(1 for _ in check.iter('MultiObjectiveGeneticAlgorithm')) >= len(names)
  return newText, names, notes


def main(argv):
  """
    Command line: convert the listed files in place (a .bak copy of each changed file is kept).
    @ In, argv, list(str), command-line arguments
    @ Out, failures, int, number of files that could not be converted
  """
  checkOnly = '--check' in argv
  files = [arg for arg in argv[1:] if arg != '--check']
  if not files:
    print(__doc__)
    return 1
  failures = 0
  for fileName in files:
    try:
      with open(fileName, 'r') as stream:
        text = stream.read()
      newText, names, notes = convertText(text)
    except (OSError, ET.ParseError, ValueError) as err:
      print(f'ERROR  {fileName}: {err}')
      failures += 1
      continue
    for note in notes:
      print(f'REVIEW {fileName}: {note}')
    if not names:
      print(f'ok     {fileName}: nothing to convert')
      continue
    if checkOnly:
      print(f'would convert {fileName}: {", ".join(names)}')
      continue
    backup = fileName + '.bak'
    if not os.path.exists(backup):
      with open(backup, 'w') as stream:
        stream.write(text)
    with open(fileName, 'w') as stream:
      stream.write(newText)
    print(f'converted {fileName}: {", ".join(names)} -> MultiObjectiveGeneticAlgorithm type="NSGA-II"')
  return failures


if __name__ == '__main__':
  sys.exit(main(sys.argv))
