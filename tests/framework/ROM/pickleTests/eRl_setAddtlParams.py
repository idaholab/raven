# Copyright 2022 Battelle Energy Alliance, LLC
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
  This test demonstrates the ability for users to change the clusterEvalMode of a ROM with the new
  setAdditionalParams method in the externalROMloader file. This method should allow the user to set any ROM
  parameters that are present in the pickledROM class; however, that is not yet demonstrated in existing tests.
  The benefit of this method is to allow users to externally evaluate pickled ROM(s) with some ability to
  customize the evaluation settings.
"""
import sys
import os
import numpy as np

# Add romLoader to path
here = os.path.abspath(os.path.dirname(__file__))
# Paths for the ravenframework and test pk file
frameworkPath = os.path.abspath(os.path.join(here, *['..']*4, 'ravenframework'))
picklePath = os.path.join(here,'ARMA', 'arma.pk')
assert os.path.exists(picklePath)
# Appending externalROMloader to path
sys.path.append(os.path.abspath(os.path.join(frameworkPath, '..', 'scripts')))
import externalROMloader

def check(runner, before, after, signal):
    """
      Decides of application of setAdditionalParams was successful
      @ In, runner, externalROMloader object
      @ In, before/after, list, list of xml objects
      @ In, signal, string, name of signal that exists in ROM
      @ Out, results, dict, counts of passes and fails
    """
    # Initializing the results index
    results = {'pass':0, 'fail':0}
    # Placeholder input necessary for evaluate method
    inp = {'scaling':[1]}
    # Running method of testing interest and evaluating
    runner.setAdditionalParams(before)
    res = runner.evaluate(inp)[0]
    beforeResult = res['_indexMap'][0][signal]
    # Checking for clustered index in non clustered eval
    if '_ROM_Cluster' not in beforeResult:
        results['pass'] += 1
    else:
        results['fail'] += 1
    runner.setAdditionalParams(after)
    res = runner.evaluate(inp)[0]
    afterResult = res['_indexMap'][0][signal]
    # Checking for clustered index in clustered eval
    if '_ROM_Cluster' in afterResult:
        results['pass'] += 1
    else:
        results['fail'] += 1
    # Returning results
    return results


def initialize(picklePath, frameworkPath):
    """
      Initializes the ravenROMexternal object and imports xmlUtils
      @ In, picklePath, path variable to pickle file in same folder as test
      @ In, frameworkPath, relative path to framework from working directory
      @ Out, runner, ROMloader object for externally evaluating ROM(s)
    """
    runner = externalROMloader.ravenROMexternal(picklePath,frameworkPath)
    return runner

def nodeGenerator(clusterEvalMode):
    """
      Defines nodes object specifically for setting clusterEvalMode
      @ In, clusterEvalMode, string, 'full' or 'clustered'
      @ Out, nodes, list, list of xml nodes for setting pickleROM parameters
    """
    #NOTE this kind of node definition should apply to any ROM parameter changes for this method
    # as long as the parameter being set with the xml node exists in the pickleROM object
    # Importing xml node tool
    from ravenframework.utils import xmlUtils
    nodes = []
    node = xmlUtils.newNode('loadedROM', attrib={'name': 'ROM', 'subType': 'pickledROM'})
    node.append(xmlUtils.newNode('clusterEvalMode', text=clusterEvalMode))
    nodes.append(node)
    return nodes

def main():
    """
      Runs the all the methods of the test script
    """
    # Initializing runner, make sure to call this before nodeGenerator
    runner = initialize(picklePath, frameworkPath)
    # Name of signal in the ROM to run test with
    signal = 'Signal'
    # Getting 'full' and 'clustered' evaluations
    before = nodeGenerator('full')
    after = nodeGenerator('clustered')
    # Running test function
    results = check(runner, before, after, signal)
    print(results)
    sys.exit(results['fail'])

if __name__ == '__main__':
    main()