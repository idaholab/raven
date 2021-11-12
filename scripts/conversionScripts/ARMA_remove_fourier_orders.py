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

def convert(tree, fileName=None):
  """
    Converts input files to be compatible with merge request #???:
    Where ARMA exists, if FourierOrders present, move into the Fourier node explicitly
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  models = simulation.find('Models')
  if models is not None:
    for child in models:
      if child.tag == 'ROM' and child.attrib['subType'] == 'ARMA':
        orderNode = child.find('FourierOrder')
        fourierNode = child.find('Fourier')
        if orderNode is not None:
          orders = [int(x) for x in orderNode.text.split(',')]
          child.remove(orderNode)
          if fourierNode is not None:
            fouriers = [float(x) for x in fourierNode.text.split(',')]
            new = []
            for i, fourier in enumerate(fouriers):
              for o in range(orders[i]):
                new.append(fourier / float(o+1))
            new = sorted(list(set(new)), reverse=True)
            fourierNode.text = ', '.join(['{:f}'.format(x) for x in new])
  return tree

if __name__ == '__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
