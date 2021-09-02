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
ndSpline = r"""
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

gaussPolynomialRom = r"""
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

hdmrRom = r"""
\hspace{24pt}
Example:
\begin{lstlisting}[style=XML,morekeywords={name,subType}]
  <Samplers>
    ...
    <Sobol name="mySobol" parallel="0">
      <variable name="x1">
        <distribution>myDist1</distribution>
      </variable>
      <variable name="x2">
        <distribution>myDist2</distribution>
      </variable>
      <ROM class = 'Models' type = 'ROM' >myHDMR</ROM>
    </Sobol>
    ...
  </Samplers>
  ...
  <Models>
    ...
    <ROM name='myHDMR' subType='HDMRRom'>
      <Target>ans</Target>
      <Features>x1,x2</Features>
      <SobolOrder>2</SobolOrder>
      <IndexSet>TotalDegree</IndexSet>
      <PolynomialOrder>4</PolynomialOrder>
      <Interpolation quad='Legendre' poly='Legendre' weight='1'>x1</Interpolation>
      <Interpolation quad='ClenshawCurtis' poly='Jacobi' weight='2'>x2</Interpolation>
    </ROM>
    ...
  </Models>
\end{lstlisting}
"""

msr = r"""
\hspace{24pt}
Example:
\begin{lstlisting}[style=XML,morekeywords={name,subType}]
<Simulation>
  ...
  <Models>
    ...
    </ROM>
    <ROM name='aUserDefinedName' subType='MSR'>
       <Features>var1,var2,var3</Features>
       <Target>result1,result2</Target>
       <!-- <weighted>true</weighted> -->
       <simplification>0.0</simplification>
       <persistence>difference</persistence>
       <gradient>steepest</gradient>
       <graph>beta skeleton</graph>
       <beta>1</beta>
       <knn>8</knn>
       <partitionPredictor>kde</partitionPredictor>
       <kernel>gaussian</kernel>
       <smooth/>
       <bandwidth>0.2</bandwidth>
     </ROM>
    ...
  </Models>
  ...
</Simulation>
\end{lstlisting}
"""

invDist = r"""
\hspace{24pt}
Example:
\begin{lstlisting}[style=XML,morekeywords={name,subType}]
<Simulation>
  ...
  <Models>
    ...
    <ROM name='aUserDefinedName' subType='NDinvDistWeight'>
      <Features>var1,var2,var3</Features>
      <Target>result1,result2</Target>
      <p>3</p>
     </ROM>
    ...
  </Models>
  ...
</Simulation>
\end{lstlisting}
"""

synthetic = r"""
\hspace{24pt}
Example:
\begin{lstlisting}[style=XML,morekeywords={name,subType,pivotLength,shift,target,threshold,period,width}]
<Simulation>
  ...
  <Models>
    ...
    <ROM name="synth" subType="SyntheticHistory">
      <Target>signal1, signal2, hour</Target>
      <Features>scaling</Features>
      <pivotParameter>hour</pivotParameter>
      <fourier target="signal1, signal2">
        <periods>12, 24</periods>
      </fourier>
      <arma target="signal1, signal2" seed='42'>
        <SignalLag>2</SignalLag>
        <NoiseLag>3</NoiseLag>
      </arma>
    </ROM>
    ...
  </Models>
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

poly = r"""
\hspace{24pt}
Example:
\begin{lstlisting}[style=XML,morekeywords={name,subType}]
<Simulation>
  ...
  <Models>
    ...
   <ROM name='PolyExp' subType='PolyExponential'>
     <Target>time,decay_heat, xe135_dens</Target>
     <Features>enrichment,bu</Features>
     <pivotParameter>time</pivotParameter>
     <numberExpTerms>5</numberExpTerms>
     <max_iter>1000000</max_iter>
     <tol>0.000001</tol>
  </ROM>
    ...
  </Models>
  ...
</Simulation>
\end{lstlisting}

Example to export the coefficients of trained PolyExponential ROM:
\begin{lstlisting}[style=XML,morekeywords={name,subType}]
<Simulation>
  ...
  <OutStreams>
    ...
    <Print name = 'dumpAllCoefficients'>
      <type>xml</type>
      <source>PolyExp</source>
      <!--
        here the <what> node is omitted. All the available params/coefficients
        are going to be printed out
      -->
    </Print>
    <Print name = 'dumpSomeCoefficients'>
      <type>xml</type>
      <source>PolyExp</source>
      <what>coefficients,timeScale</what>
    </Print>
    ...
  </OutStreams>
  ...
</Simulation>
\end{lstlisting}
"""

dmd = r"""
\hspace{24pt}
Example:
\textbf{Example:}
\begin{lstlisting}[style=XML,morekeywords={name,subType}]
<Simulation>
  ...
  <Models>
    ...
   <ROM name='DMD' subType='DMD'>
      <Target>time,totals_watts, xe135_dens</Target>
      <Features>enrichment,bu</Features>
      <dmdType>dmd</dmdType>
      <pivotParameter>time</pivotParameter>
      <rankSVD>0</rankSVD>
      <rankTLSQ>5</rankTLSQ>
      <exactModes>False</exactModes>
      <optimized>True</optimized>
    </ROM
    ...
  </Models>
  ...
</Simulation>
\end{lstlisting}

Example to export the coefficients of trained DMD ROM:
\begin{lstlisting}[style=XML,morekeywords={name,subType}]
<Simulation>
  ...
  <OutStreams>
    ...
    <Print name = 'dumpAllCoefficients'>
      <type>xml</type>
      <source>DMD</source>
      <!--
        here the <what> node is omitted. All the available params/coefficients
        are going to be printed out
      -->
    </Print>
    <Print name = 'dumpSomeCoefficients'>
      <type>xml</type>
      <source>PolyExp</source>
      <what>eigs,amplitudes,modes</what>
    </Print>
    ...
  </OutStreams>
  ...
</Simulation>
\end{lstlisting}
"""

kmlpc = r"""
\hspace{24pt}
Example:
\begin{lstlisting}[style=XML,morekeywords={name,subType}]
<Simulation>
  ...
  <Models>
    ...
    <ROM name='aUserDefinedName' subType='KerasMLPClassifier'>
      <Features>X,Y</Features>
      <Target>Z</Target>
      <loss>mean_squared_error</loss>
      <metrics>accuracy</metrics>
      <batch_size>4</batch_size>
      <epochs>4</epochs>
      <optimizerSetting>
        <beta_1>0.9</beta_1>
        <optimizer>Adam</optimizer>
        <beta_2>0.999</beta_2>
        <epsilon>1e-8</epsilon>
        <decay>0.0</decay>
        <lr>0.001</lr>
      </optimizerSetting>
      <Dense name="layer1">
          <activation>relu</activation>
          <dim_out>15</dim_out>
      </Dense>
      <Dropout name="dropout1">
          <rate>0.2</rate>
      </Dropout>
      <Dense name="layer2">
          <activation>tanh</activation>
          <dim_out>8</dim_out>
      </Dense>
      <Dropout name="dropout2">
          <rate>0.2</rate>
      </Dropout>
      <Dense name="outLayer">
          <activation>sigmoid</activation>
      </Dense>
      <layer_layout>layer1, dropout1, layer2, dropout2, outLayer</layer_layout>
    </ROM>
    ...
  </Models>
  ...
</Simulation>
\end{lstlisting}
"""

kconv = r"""
\hspace{24pt}
Example:
\begin{lstlisting}[style=XML,morekeywords={name,subType}]
<Simulation>
  ...
  <Models>
    ...
    <ROM name='aUserDefinedName' subType='KerasConvNetClassifier'>
      <Features>x1,x2</Features>
      <Target>labels</Target>
      <loss>mean_squared_error</loss>
      <metrics>accuracy</metrics>
      <batch_size>1</batch_size>
      <epochs>2</epochs>
      <plot_model>True</plot_model>
      <validation_split>0.25</validation_split>
      <num_classes>1</num_classes>
      <optimizerSetting>
        <beta_1>0.9</beta_1>
        <optimizer>Adam</optimizer>
        <beta_2>0.999</beta_2>
        <epsilon>1e-8</epsilon>
        <decay>0.0</decay>
        <lr>0.001</lr>
      </optimizerSetting>
      <Conv1D name="firstConv1D">
          <activation>relu</activation>
          <strides>1</strides>
          <kernel_size>2</kernel_size>
          <padding>valid</padding>
          <dim_out>32</dim_out>
      </Conv1D>
      <MaxPooling1D name="pooling1">
          <strides>2</strides>
          <pool_size>2</pool_size>
      </MaxPooling1D>
      <Conv1D name="SecondConv1D">
          <activation>relu</activation>
          <strides>1</strides>
          <kernel_size>2</kernel_size>
          <padding>valid</padding>
          <dim_out>32</dim_out>
      </Conv1D>
      <MaxPooling1D name="pooling2">
          <strides>2</strides>
          <pool_size>2</pool_size>
      </MaxPooling1D>
      <Flatten name="flatten">
      </Flatten>
      <Dense name="dense1">
          <activation>relu</activation>
          <dim_out>10</dim_out>
      </Dense>
      <Dropout name="dropout1">
          <rate>0.25</rate>
      </Dropout>
      <Dropout name="dropout2">
          <rate>0.25</rate>
      </Dropout>
      <Dense name="dense2">
          <activation>softmax</activation>
      </Dense>
      <layer_layout>firstConv1D, pooling1, dropout1, SecondConv1D, pooling2, dropout2, flatten, dense1, dense2</layer_layout>
    </ROM>
    ...
  </Models>
  ...
</Simulation>
\end{lstlisting}
"""

klstmc = r"""
\hspace{24pt}
\textbf{KerasLSTMClassifier Example:}
\begin{lstlisting}[style=XML,morekeywords={name,subType}]
<Simulation>
  ...
  <Models>
    ...
    <ROM name='aUserDefinedName' subType='KerasLSTMClassifier'>
      <Features>x</Features>
      <Target>y</Target>
      <loss>categorical_crossentropy</loss>
      <metrics>accuracy</metrics>
      <batch_size>1</batch_size>
      <epochs>10</epochs>
      <validation_split>0.25</validation_split>
      <num_classes>26</num_classes>
      <optimizerSetting>
        <beta_1>0.9</beta_1>
        <optimizer>Adam</optimizer>
        <beta_2>0.999</beta_2>
        <epsilon>1e-8</epsilon>
        <decay>0.0</decay>
        <lr>0.001</lr>
      </optimizerSetting>
      <LSTM name="lstm1">
          <activation>tanh</activation>
          <dim_out>32</dim_out>
      </LSTM>
      <LSTM name="lstm2">
          <activation>tanh</activation>
          <dim_out>16</dim_out>
      </LSTM>
      <Dropout name="dropout">
          <rate>0.25</rate>
      </Dropout>
      <Dense name="dense">
          <activation>softmax</activation>
      </Dense>
      <layer_layout>lstm1,lstm2,dropout,dense</layer_layout>
    </ROM>
    ...
  </Models>
  ...
</Simulation>
\end{lstlisting}

"""

klstmr = r"""
\hspace{24pt}
\textbf{KerasLSTMRegression Example:}
\begin{lstlisting}[style=XML,morekeywords={name,subType}]
<Simulation>
  ...
  <Models>
    ...
    <ROM name="lstmROM" subType="KerasLSTMRegression">
      <Features>prev_sum, prev_square, prev_square_sum</Features>
      <Target>sum, square</Target>
      <pivotParameter>index</pivotParameter>
      <loss>mean_squared_error</loss>
      <LSTM name="lstm1">
        <dim_out>32</dim_out>
      </LSTM>
      <LSTM name="lstm2">
        <dim_out>16</dim_out>
      </LSTM>
      <Dense name="dense">
      </Dense>
      <layer_layout>lstm1, lstm2, dense</layer_layout>

    </ROM>
    ...
  </Models>
  ...
</Simulation>
\end{lstlisting}
"""

# examples Factory
exampleFactory = {
                  'NDspline':ndSpline,
                  'pickledROM':pickledROM,
                  'GaussPolynomialRom': gaussPolynomialRom,
                  'HDMRRom': hdmrRom,
                  'MSR': msr,
                  'NDinvDistWeight':invDist,
                  'SyntheticHistory': synthetic,
                  'ARMA': armaExp,
                  'PolyExponential': poly,
                  'DMD': dmd,
                  'KerasMLPClassifier': kmlpc,
                  'KerasConvNetClassifier': kconv,
                  'KerasLSTMClassifier': klstmc,
                  'KerasLSTMRegression': klstmr
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
from SupervisedLearning import ScikitLearnBase
excludeObj = ['SupervisedLearning',
              'ScikitLearnBase',
              'KerasBase',
              'KerasRegression',
              'KerasClassifier',
              'NDinterpolatorRom',
              'Collection',
              'Segments',
              'Clusters',
              'Interpolated']
validDNNRom = ['KerasMLPClassifier',
              'KerasConvNetClassifier',
              'KerasLSTMClassifier',
              'KerasLSTMRegression']
validInternalRom = ['NDspline',
            'pickledROM',
            'GaussPolynomialRom',
            'HDMRRom',
            'MSR',
            'NDinvDistWeight',
            'SyntheticHistory',
            'ARMA',
            'PolyExponential',
            'DMD']
validRom = list(SupervisedLearning.factory.knownTypes())
orderedValidRom = []
for rom in validInternalRom + validRom:
  if rom not in orderedValidRom:
    orderedValidRom.append(rom)
### Internal ROM file generation
internalRom = ''
sklROM = ''
dnnRom = ''
for name in orderedValidRom:
  if name in excludeObj:
    continue
  if name in validDNNRom:
    continue
  obj = SupervisedLearning.factory.returnClass(name)
  specs = obj.getInputSpecification()
  tex = specs.generateLatex(sectionLevel=2)
  exampleTex = exampleFactory[name] if name in exampleFactory else ''
  try:
    if isinstance(obj(), ScikitLearnBase):
      sklROM += tex
      sklROM += exampleTex
    else:
      internalRom += tex
      internalRom += exampleTex
  except:
    print('Can not generate latex file for ' + name)

fName = os.path.abspath(os.path.join(os.path.dirname(__file__), 'internalRom.tex'))
with open(fName, 'w') as f:
  f.writelines(internalRom)
print(f'\nSuccessfully wrote "{fName}"')

fName = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sklRom.tex'))
with open(fName, 'w') as f:
  f.writelines(sklROM)
print(f'\nSuccessfully wrote "{fName}"')
