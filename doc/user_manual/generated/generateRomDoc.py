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
ravenDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ravenDir)
try:
  import ravenframework.Driver
except Exception as e:
  print('\nWe did not find the modules needed for RAVEN; maybe the conda env is not activated?')
  raise e
sys.path.pop()

from ravenframework.utils.InputData import wrapText

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
        <P>2</P>
        <Q>3</Q>
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
      <source>DMD</source>
      <what>eigs,amplitudes,modes</what>
    </Print>
    ...
  </OutStreams>
  ...
</Simulation>
\end{lstlisting}
"""

dmdc = r"""
\hspace{24pt}
Example of DMDc ROM definition, with 1 actuator variable (u1), 3 state variables (x1, x2, x3), 2 output variables (y1, y2), and 2 scheduling parameters (mod, flow):
\begin{lstlisting}[style=XML,morekeywords={name,subType}]
<Simulation>
   ...
   <Models>
     ...
    <ROM name="DMDrom" subType="DMDC">
      <!-- Target contains Time, StateVariable Names (x) and OutputVariable Names (y) in training data -->
      <Target>Time,x1,x2,x3,y1,y2</Target>
      <!-- Actuator Variable Names (u) -->
      <actuators>u1</actuators>
      <!-- StateVariables Names (x) -->
      <stateVariables>x1,x2,x3</stateVariables>
      <!-- Pivot variable (e.g. Time) -->
      <pivotParameter>Time</pivotParameter>
      <!-- rankSVD: -1 = No truncation; 0 = optimized truncation; pos. int = truncation level -->
      <rankSVD>1</rankSVD>
      <!-- SubtractNormUXY: True = will subtract the initial values from U,X,Y -->
      <subtractNormUXY>True</subtractNormUXY>

      <!-- Features are the variable names for predictions: Actuator "u", scheduling parameters, and initial states -->
      <Features>u1,mod,flow,x1_init,x2_init,x3_init</Features>
      <!-- Initialization Variables-->
      <initStateVariables>
        x1_init,x2_init,x3_init
      </initStateVariables>
    </ROM>
     ...
   </Models>
   ...
 </Simulation>

\end{lstlisting}

Example to export the coefficients of trained DMDC ROM:
\begin{lstlisting}[style=XML,morekeywords={name,subType}]
<Simulation>
  ...
  <OutStreams>
    ...
    <Print name = 'dumpAllCoefficients'>
      <type>xml</type>
      <source>DMDc</source>
      <!--
        here the <what> node is omitted. All the available params/coefficients
        are going to be printed out
      -->
    </Print>
    <Print name = 'dumpSomeCoefficients'>
      <type>xml</type>
      <source>DMDc</source>
      <what>rankSVD,UNorm,XNorm,XLast,Atilde,Btilde</what>
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
                  'DMDC': dmdc,
                  'KerasMLPClassifier': kmlpc,
                  'KerasConvNetClassifier': kconv,
                  'KerasLSTMClassifier': klstmc,
                  'KerasLSTMRegression': klstmr
                  }

#------------#
# ROM #
#------------#

segmentTex = r"""
In addition, \xmlNode{Segment} can be used to divided the ROM. In order to enable the segmentation, the
user need to specify following information for \xmlNode{Segment}:
\begin{itemize}
  \item \xmlNode{Segment}, \xmlDesc{node, optional}, provides an alternative way to build the ROM. When
    this mode is enabled, the subspace of the ROM (e.g. ``time'') will be divided into segments as
    requested, then a distinct ROM will be trained on each of the segments. This is especially helpful if
    during the subspace the ROM representation of the signal changes significantly. For example, if the signal
    is different during summer and winter, then a signal can be divided and a distinct ROM trained on the
    segments. By default, no segmentation occurs.

    To futher enable clustering of the segments, the \xmlNode{Segment} has the following attributes:
    \begin{itemize}
      \item \xmlAttr{grouping}, \xmlDesc{string, optional field} enables the use of ROM subspace clustering in
        addition to segmenting if set to \xmlString{cluster}. If set to \xmlString{segment}, then performs
        segmentation without clustering. If clustering, then an additional node needs to be included in the
        \xmlNode{Segment} node, as described below.
        \default{segment}
    \end{itemize}

    This node takes the following subnodes:
    \begin{itemize}
      \item \xmlNode{subspace}, \xmlDesc{string, required field} designates the subspace to divide. This
        should be the pivot parameter (often ``time'') for the ROM. This node also requires an attribute
        to determine how the subspace is divided, as well as other attributes, described below:
        \begin{itemize}
          \item \xmlAttr{pivotLength}, \xmlDesc{float, optional field}, provides the value in the subspace
            that each segment should attempt to represent, independently of how the data is stored. For
            example, if the subspace has hourly resolution, is measured in seconds, and the desired
            segmentation is daily, the \xmlAttr{pivotLength} would be 86400.
            Either this option or \xmlAttr{divisions} must be provided.
          \item \xmlAttr{divisions}, \xmlDesc{integer, optional field}, as an alternative to
            \xmlAttr{pivotLength}, this attribute can be used to specify how many data points to include in
            each subdivision, rather than use the pivot values. The algorithm will attempt to split the data
            points as equally as possible.
            Either this option or \xmlAttr{pivotLength} must be provided.
          \item \xmlAttr{shift}, \xmlDesc{string, optional field}, governs the way in which the subspace is
            treated in each segment. By default, the subspace retains its actual values for each segment; for
            example, if each segment is 4 hours long, the first segment starts at time 0, the second at 4
            hours, the third at 8 hours, and so forth. Options to change this behavior are \xmlString{zero}
            and \xmlString{first}. In the case of \xmlString{zero}, each segment restarts the pivot with the
            subspace value as 0, shifting all other values similarly. In the example above, the first segment
            would start at 0, the second at 0, and the third at 0, with each ending at 4 hours. Note that the
            pivot values are restored when the ROM is evaluated. Using \xmlString{first}, each segment
            subspace restarts at the value of the first segment. This is useful in the event subspace 0 is not
            a desirable value.
        \end{itemize}
      \item \xmlNode{Classifier}, \xmlDesc{string, optional field} associates a \xmlNode{PostProcessor}
        defined in the \xmlNode{Models} block to this segmentation. If clustering is enabled (see
        \xmlAttr{grouping} above), then this associated Classifier will be used to cluster the segmented ROM
        subspaces. The attributes \xmlAttr{class}=\xmlString{Models} and
        \xmlAttr{type}=\xmlString{PostProcessor} must be set, and the text of this node is the \xmlAttr{name}
        of the requested Classifier. Note this Classifier must be a valid Classifier; not all PostProcessors
        are suitable. For example, see the DataMining PostProcessor subtype Clustering.
      \item \xmlNode{clusterFeatures}, \xmlDesc{string, optional field}, if clustering then delineates
        the fundamental ROM features that should be considered while clustering. The available features are
        ROM-dependent, and an exception is raised if an unrecognized request is given. See individual ROMs
        for options. \default All ROM-specific options.
      \item \xmlNode{evalMode}, \xmlDesc{string, optional field}, one of \xmlString{truncated},
        \xmlString{full}, or \xmlString{clustered}, determines how the evaluations are
        represented, as follows:
        \begin{itemize}
          \item \xmlString{full}, reproduce the full signal using representative cluster segments,
          \item \xmlString{truncated}, reproduce a history containing exactly segment from each
            cluster placed back-to-back, with the \xmlNode{pivotParameter} spanning the clustered
            dimension. Note this will almost surely not be the same length as the original signal;
            information about indexing can be found in the ROM's XML metadata.
          \item \xmlString{clustered}, reproduce a N-dimensional object with the variable
            \texttt{\_ROM\_cluster} as one of the indexes for the ROM's sampled variables. Note that
            in order to use the option, the receiving \xmlNode{DataObject} should be of type
            \xmlNode{DataSet} with one of the indices being \texttt{\_ROM\_cluster}.
        \end{itemize}
     \item \xmlNode{evaluationClusterChoice}, \xmlDesc{string, optional field}, one of \xmlString{first} or
        \xmlString{random}, determines, if \xmlAttr{grouping}$=cluster$, which
        strategy needs to be followed for the evaluation stage. If ``first'', the
        first ROM (representative segmented ROM),in each cluster, is considered to
         be representative of the full space in the cluster (i.e. the evaluation is always performed
         interrogating the first ROM in each cluster); If ``random'', a random ROM, in each cluster,
         is choosen when an evaluation is requested.
   \nb if ``first'' is used, there is \emph{substantial} memory savings when compared to using
   ``random''.
         %If ``centroid'', a ROM ``trained" on the centroids
         %information of each cluster is used for the evaluation (\nb ``centroid'' option is not
         %available yet).
         \default{first}
    \end{itemize}
\end{itemize}
"""

from ravenframework import SupervisedLearning
from ravenframework.SupervisedLearning import ScikitLearnBase
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
              'KerasMLPRegression',
              'KerasConvNetClassifier',
              'KerasLSTMClassifier',
              'KerasLSTMRegression'
              ]
validInternalRom = ['NDspline',
            'pickledROM',
            'GaussPolynomialRom',
            'HDMRRom',
            'MSR',
            'NDinvDistWeight',
            'SyntheticHistory',
            'ARMA',
            'PolyExponential',
            'DMD',
            'DMDC']
validRom = list(SupervisedLearning.factory.knownTypes())
orderedValidRom = []
for rom in validInternalRom + validRom:
  if rom not in orderedValidRom:
    orderedValidRom.append(rom)
### Internal ROM file generation
internalRom = ''
sklROM = ''
dnnRom = ''
# base classes first
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
      if name == 'ARMA':
        internalRom += segmentTex
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
