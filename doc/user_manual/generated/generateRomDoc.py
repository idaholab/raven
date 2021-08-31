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
pickledROM = r"""
\hspace{24pt}
Example:
For this example the ROM has already been created and trained in another RAVEN run, then pickled to a file
called \texttt{rom\_pickle.pk}.  In the example, the file is identified in \xmlNode{Files}, the model is
defined in \xmlNode{Models}, and the model loaded in \xmlNode{Steps}.
\begin{lstlisting}[style=XML]
<Simulation>
  ...
  <Files>
    <Input name="rompk" type="">rom_pickle.pk</Input>
  </Files>
  ...
  <Models>
    ...
    <ROM name="myRom" subType="pickledROM"/>
    ...
  </Models>
  ...
  <Steps>
    ...
    <IOStep name="loadROM">
      <Input class="Files" type="">rompk</Input>
      <Output class="Models" type="ROM">myRom</Output>
    </IOStep>
    ...
  </Steps>
  ...
</Simulation>
\end{lstlisting}
"""

armaExp = r"""
\hspace{24pt}
General ARMA Example:
\begin{lstlisting}[style=XML, morekeywords={name,subType,pivotLength,shift,target,threshold,period,width}]
<Simulation>
  ...
  <Models>
    ...
    <ROM name='aUserDefinedName' subType='ARMA'>
      <pivotParameter>Time</pivotParameter>
      <Features>scaling</Features>
      <Target>Speed1,Speed2</Target>
      <P>5</P>
      <Q>4</Q>
      <Segment>
        <subspace pivotLength="1296000" shift="first">Time</subspace>
      </Segment>
      <preserveInputCDF>True</preserveInputCDF>
      <Fourier>604800,86400</Fourier>
      <FourierOrder>2, 4</FourierOrder>
      <Peaks target='Speed1' threshold='0.1' period='86400'>
        <window width='14400' >-7200,10800</window>
        <window width='18000' >64800,75600</window>
      </Peaks>
     </ROM>
    ...
  </Models>
  ...
</Simulation>
\end{lstlisting}
"""

NDspline = r"""
\hspace{24pt}
Example:
\begin{lstlisting}[style=XML]
<Simulation>
  ...
  <Models>
    ...
    <ROM name='aUserDefinedName' subType='NDspline'>
       <Features>var1,var2,var3</Features>
       <Target>result1,result2</Target>
     </ROM>
    ...
  </Models>
  ...
</Simulation>
\end{lstlisting}
"""

GaussPolynomialRom = r"""
\hspace{24pt}
Example:
\begin{lstlisting}[style=XML,morekeywords={name,subType}]
<Simulation>
  ...
  <Samplers>
    ...
    <SparseGridCollocation name="mySG" parallel="0">
      <variable name="x1">
        <distribution>myDist1</distribution>
      </variable>
      <variable name="x2">
        <distribution>myDist2</distribution>
      </variable>
      <ROM class = 'Models' type = 'ROM' >myROM</ROM>
    </SparseGridCollocation>
    ...
  </Samplers>
  ...
  <Models>
    ...
    <ROM name='myRom' subType='GaussPolynomialRom'>
      <Target>ans</Target>
      <Features>x1,x2</Features>
      <IndexSet>TotalDegree</IndexSet>
      <PolynomialOrder>4</PolynomialOrder>
      <Interpolation quad='Legendre' poly='Legendre' weight='1'>x1</Interpolation>
      <Interpolation quad='ClenshawCurtis' poly='Jacobi' weight='2'>x2</Interpolation>
    </ROM>
    ...
  </Models>
  ...
</Simulation>
\end{lstlisting}
"""

rom = r"""
\hspace{24pt}
Example:
\begin{lstlisting}[style=XML]

\end{lstlisting}
"""

rom = r"""
\hspace{24pt}
Example:
\begin{lstlisting}[style=XML]

\end{lstlisting}
"""

rom = r"""
\hspace{24pt}
Example:
\begin{lstlisting}[style=XML]

\end{lstlisting}
"""

rom = r"""
\hspace{24pt}
Example:
\begin{lstlisting}[style=XML]

\end{lstlisting}
"""

rom = r"""
\hspace{24pt}
Example:
\begin{lstlisting}[style=XML]

\end{lstlisting}
"""

rom = r"""
\hspace{24pt}
Example:
\begin{lstlisting}[style=XML]

\end{lstlisting}
"""


# examples Factory
exampleFactory = {'pickledROM':pickledROM,
                  'ARMA':armaExp,
                  'NDspline':NDspline,
                  'GaussPolynomialRom': GaussPolynomialRom,

                  }

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
excludeObj = ['SupervisedLearning', 'ScikitLearnBase', 'KerasBase', 'KerasRegression', 'KerasClassifier',
              'Collection', 'Segments', 'Clusters', 'Interpolated']

validRom = ['NDspline',
            'pickledROM',
            'GaussPolynomialRom',

            ]
# write all known types
for name in SupervisedLearning.factory.knownTypes():
  if name in excludeObj:
    continue
  obj = SupervisedLearning.factory.returnClass(name)
  specs = obj.getInputSpecification()
  tex = specs.generateLatex()
  msg +=tex
  if name in exampleFactory:
    msg+= exampleFactory[name]

fName = os.path.abspath(os.path.join(os.path.dirname(__file__), 'rom.tex'))
with open(fName, 'w') as f:
  f.writelines(msg)

print(f'\nSuccessfully wrote "{fName}"')
