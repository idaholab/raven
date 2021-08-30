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
  Created on Aug. 30, 2021
  @author: wangc
  Generates documentation for ROM
"""

# get driver and path and import it, to get RAVEN paths correct
import os
import sys
ravenFramework = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'framework'))
sys.path.append(ravenFramework)
try:
  import Driver
except Exception as e:
  print('\nWe did not find the modules needed for RAVEN; maybe the conda env is not activated?')
  raise e
sys.path.pop()

from utils.InputData import wrapText

# examples
romExample = r"""
\hspace{24pt}
Example:
\begin{lstlisting}[style=XML]
<ROM name='rom' subType='SVR'>
  ...

  ...
</ROM>
\end{lstlisting}

"""


# examples Factory
exampleFactory = {'romExample':romExample}

#------------#
# ROM #
#------------#
import Models
msg = ''
# base classes first
# descr = wrapText(Models.ROM.userManualDescription(), '  ')
descr = ' '
msg += descr

import SupervisedLearning
# write all known types
for name in SupervisedLearning.factory.knownTypes():
  obj = SupervisedLearning.factory.returnClass(name)
  specs = obj.getInputSpecification()
  tex = specs.generateLatex()
  msg +=tex
  # msg+= exampleFactory[name]

fName = os.path.abspath(os.path.join(os.path.dirname(__file__), 'rom.tex'))
with open(fName, 'w') as f:
  f.writelines(msg)

print(f'\nSuccessfully wrote "{fName}"')
