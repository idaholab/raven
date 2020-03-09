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
  Generates documentation from base classes. WIP.
   - talbpw, 2020
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

from MessageHandler import MessageHandler, MessageUser
# message handler
mh = MessageHandler()
mu = MessageUser()
mu.messageHandler = mh

#------------#
# OPTIMIZERS #
#------------#
import Optimizers
all = ''
# base classes first
optDescr = wrapText(Optimizers.Optimizer.userManualDescription(), '  ')
all += optDescr
# write all known types
for name in Optimizers.knownTypes():
  obj = Optimizers.returnClass(name, mu)
  specs = obj.getInputSpecification()
  tex = specs.generateLatex()
  all += tex

# examples
minimal = r"""
\hspace{24pt}
Gradient Descent Example:
\begin{lstlisting}[style=XML]
<Optimizers>
  ...
  <GradientDescent name="opter">
    <objective>ans</objective>
    <variable name="x">
      <distribution>x_dist</distribution>
      <initial>-2</initial>
    </variable>
    <variable name="y">
      <distribution>y_dist</distribution>
      <initial>2</initial>
    </variable>
    <samplerInit>
      <limit>100</limit>
    </samplerInit>
    <gradient>
      <FiniteDifference/>
    </gradient>
    <stepSize>
      <GradientHistory/>
    </stepSize>
    <acceptance>
      <Strict/>
    </acceptance>
    <TargetEvaluation class="DataObjects" type="PointSet">optOut</TargetEvaluation>
  </GradientDescent>
  ...
</Optimizers>
\end{lstlisting}

"""
all += minimal
fName = os.path.abspath(os.path.join(os.path.dirname(__file__), 'optimizer.tex'))
with open(fName, 'w') as f:
  f.writelines(all)
