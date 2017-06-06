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
from __future__ import print_function
import sys, os, time
import xml.etree.ElementTree as ET
#load XML navigation tools
sys.path.append(os.path.join(os.getcwd(),'..','..','framework'))
from utils import xmlUtils

def getNode(fname,nodepath):
  """
    Searches file for particular node.  Note that this takes the first matching node path in the file.
    @ In, fname, string, name of file with XML tree to search
    @ In, nodepath, string, "."-separated string with xml node name path, i.e., Simulation.RunInfo.Sequence
    @ Out, getNode, ET.Element, prettified element
  """
  #TODO add option to include parent nodes with ellipses
  root = ET.parse(fname).getroot()
  #format nodepath
  nodepath = nodepath.replace('.','|')
  #check if root is desired node
  if root.tag == nodepath:
    node = root
    docLevel = 0
  else:
    docLevel = len(nodepath.split('|'))
    node = xmlUtils.findPathEllipsesParents(root,nodepath,docLevel)
    if node is None:
      raise IOError('Unable to find '+nodepath+' in '+fname)
  return xmlUtils.prettify(node,doc=True,docLevel=docLevel)


def chooseSize(string):
  """
    Determines appropriate latex size to use so that the string fits in RAVEN-standard lstlisting examples
      without flowing over to newline.  Could be improved to consider the overall number of lines in the
      string as well, so that we don't have multi-page examples very often.
    @ In, string, string, the string by which to determine the size
    @ Out, chooseSize, string, the LaTeX-ready size at which the string should be represented
  """
  longest = 0
  for line in string.split('\n'):
    longest = max(longest,len(line.rstrip()))
  if longest < 64:
    return '\\normalsize'
  elif longest < 70:
    return '\\small'
  elif longest < 77:
    return '\\footnotesize'
  elif longest < 96:
    return '\\scriptsize'
  else:
    return '\\tiny'


if __name__=='__main__':
  #if len(sys.argv)<5 or '-f' not in sys.argv or '-n' not in sys.argv:
  try:
    fname = sys.argv[sys.argv.index('-f')+1]
    nname = sys.argv[sys.argv.index('-n')+1]
    if '-h' in sys.argv:
      highlight = sys.argv[sys.argv.index('-h')+1]
    else:
      highlight = None #more difficult to re-parse
  except IndexError:
    print('Error calling writeTex.py!\n  Syntax is: python writeTex.py -f /path/to/file.xml -n name_of_node -h additional,keywords')
    raise IOError()
  #re-parse filename and path to be system dependent
  fname = os.path.join(*fname.split('/'))
  print('reading dynamic XML from',fname)
  strNode = getNode(fname,nname)
  outFile = file('raven_temp_tex_xml.tex','w')
  printName = os.path.join('raven',fname.replace('../','')).replace('_','\_')
  toWrite = '\\FloatBarrier\n'+\
            chooseSize(printName)+'\n\\texttt{'+printName+'}\n'+\
            chooseSize(strNode)+'\n'+\
            '\\begin{lstlisting}[style=XML'
  if highlight is not None:
    toWrite += ',morekeywords={'+highlight+'}'
  toWrite += ']\n'+\
             strNode+'\n'+\
             '\\end{lstlisting}\n'+\
             '\\normalsize'
  outFile.writelines(toWrite)
  outFile.close()

