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
  Tests for scripts/conversionScripts/toMultiObjectiveGA.py, which upgrades genetic-algorithm inputs
  written before the GA refactor: multi-objective <GeneticAlgorithm> becomes
  <MultiObjectiveGeneticAlgorithm type="NSGA-II">, only the changed tags are edited (formatting and
  comments are kept byte for byte), and settings NSGA-II rejects or ignores are reported.

  @author: Mohammad Abdo (@Jimmy-INL)
"""
import os
import sys
import xml.etree.ElementTree as ET

import pytest

RAVEN = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4))
sys.path.insert(0, os.path.join(RAVEN, 'scripts', 'conversionScripts'))
import toMultiObjectiveGA as converter

OLD_MULTI = """<?xml version="1.0" ?>
<Simulation verbosity="debug">
  <Optimizers>
    <!-- a comment with <angle> & ampersand characters -->
    <GeneticAlgorithm name='GAopt'>
      <objective>MaxEFPD, FuelCycleCost</objective>

      <ImplicitConstraint class='Functions' type='External'>impConstr1</ImplicitConstraint>
      <GAparams>
        <parentSelection>rouletteWheel</parentSelection>
        <fitness type="feasibleFirst">
          <b>0.0025</b>
        </fitness>
        <survivorSelection>fitnessBased</survivorSelection>
      </GAparams>
    </GeneticAlgorithm>
    <GeneticAlgorithm name="single">
      <objective>ans</objective>
      <GAparams>
        <fitness type="invLinear"/>
      </GAparams>
    </GeneticAlgorithm>
  </Optimizers>
  <Steps>
    <MultiRun name="opt">
      <Optimizer      class="Optimizers"  type="GeneticAlgorithm" >GAopt</Optimizer>
    </MultiRun>
    <MultiRun name="other">
      <Optimizer class="Optimizers" type="GeneticAlgorithm">single</Optimizer>
    </MultiRun>
  </Steps>
</Simulation>
"""


def _lineDiff(a, b):
  """
    Lines that differ between two texts.
    @ In, a, str, first text
    @ In, b, str, second text
    @ Out, diff, list, (old, new) line pairs
  """
  la, lb = a.splitlines(), b.splitlines()
  assert len(la) == len(lb), 'the conversion must not add or remove lines'
  return [(x, y) for x, y in zip(la, lb) if x != y]


def testMultiObjectiveConvertedAndStepReferenceUpdated():
  newText, names, _ = converter.convertText(OLD_MULTI)
  assert names == ['GAopt']
  root = ET.fromstring(newText)
  opt = root.find('Optimizers/MultiObjectiveGeneticAlgorithm')
  assert opt.attrib == {'name': 'GAopt', 'type': 'NSGA-II'}
  refs = {ref.text.strip(): ref.attrib['type'] for ref in root.iter('Optimizer')}
  assert refs == {'GAopt': 'MultiObjectiveGeneticAlgorithm', 'single': 'GeneticAlgorithm'}


def testOnlyChangedTagsAreEdited():
  """Comments, quote style, blank lines and indentation are preserved; exactly three lines change."""
  newText, _, _ = converter.convertText(OLD_MULTI)
  diff = _lineDiff(OLD_MULTI, newText)
  assert diff == [
    ("    <GeneticAlgorithm name='GAopt'>", "    <MultiObjectiveGeneticAlgorithm name='GAopt' type=\"NSGA-II\">"),
    ('    </GeneticAlgorithm>', '    </MultiObjectiveGeneticAlgorithm>'),
    ('      <Optimizer      class="Optimizers"  type="GeneticAlgorithm" >GAopt</Optimizer>',
     '      <Optimizer      class="Optimizers"  type="MultiObjectiveGeneticAlgorithm" >GAopt</Optimizer>'),
  ]


def testSingleObjectiveUntouched():
  text = OLD_MULTI.replace('MaxEFPD, FuelCycleCost', 'MaxEFPD')
  newText, names, _ = converter.convertText(text)
  assert names == [] and newText == text


def testReviewNotes():
  """Settings that NSGA-II rejects, and a fitness that no longer drives selection, are reported."""
  _, _, notes = converter.convertText(OLD_MULTI)
  joined = '\n'.join(notes)
  assert 'rouletteWheel' in joined and 'tournamentSelection' in joined
  assert 'fitnessBased' in joined and 'rankNcrowdingBased' in joined
  assert 'feasibleFirstPenalty' in joined


def testSingleObjectiveWithoutFitnessReported():
  text = OLD_MULTI.replace('        <fitness type="invLinear"/>\n', '')
  _, _, notes = converter.convertText(text)
  assert any('single' in note and 'no <fitness>' in note for note in notes)


def testExistingTypeAttributeKept():
  text = OLD_MULTI.replace("<GeneticAlgorithm name='GAopt'>", "<GeneticAlgorithm name='GAopt' type='NSGA-II'>")
  newText, _, _ = converter.convertText(text)
  assert newText.count('NSGA-II') == 1


def testCommandLineWritesBackupAndCheckWritesNothing(tmp_path):
  deck = tmp_path / 'deck.xml'
  deck.write_text(OLD_MULTI)
  assert converter.main(['toMultiObjectiveGA.py', '--check', str(deck)]) == 0
  assert deck.read_text() == OLD_MULTI and not (tmp_path / 'deck.xml.bak').exists()
  assert converter.main(['toMultiObjectiveGA.py', str(deck)]) == 0
  assert (tmp_path / 'deck.xml.bak').read_text() == OLD_MULTI
  assert '<MultiObjectiveGeneticAlgorithm' in deck.read_text()
  # running again is a no-op
  converted = deck.read_text()
  assert converter.main(['toMultiObjectiveGA.py', str(deck)]) == 0
  assert deck.read_text() == converted


def testUnparseableFileReportedNotRaised(tmp_path):
  deck = tmp_path / 'bad.xml'
  deck.write_text('<Simulation><Optimizers>')
  assert converter.main(['toMultiObjectiveGA.py', str(deck)]) == 1


def testRealMultiObjectiveDeckRoundTrip():
  """A real NSGA-II regression deck, written back in the pre-refactor form, converts to the deck itself."""
  deck = os.path.join(RAVEN, 'tests', 'framework', 'Optimizers', 'GeneticAlgorithms', 'continuous',
                      'unconstrained', 'MultiObjectiveBeale-Bealeflipped.xml')
  current = open(deck).read()
  assert '<MultiObjectiveGeneticAlgorithm' in current
  # rebuild the old form: plain <GeneticAlgorithm>, no type attribute, old <Steps> reference
  old = (current.replace('<MultiObjectiveGeneticAlgorithm', '<GeneticAlgorithm')
                .replace('</MultiObjectiveGeneticAlgorithm>', '</GeneticAlgorithm>')
                .replace('type="MultiObjectiveGeneticAlgorithm"', 'type="GeneticAlgorithm"'))
  old = old.replace(' type="NSGA-II"', '', 1)
  newText, names, _ = converter.convertText(old)
  assert names and newText == current


if __name__ == '__main__':
  sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
