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
ravenDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ravenDir)
try:
  import ravenframework.Driver
except Exception as e:
  print('\nWe did not find the modules needed for RAVEN; maybe the conda env is not activated?')
  raise e
sys.path.pop()

from ravenframework.utils.InputData import wrapText

def insertSolnExport(tex, obj):
  """
    Inserts solution export blurb into tex.
    @ In, tex, str, LaTeX
    @ In, obj, object, identity being written
    @ Out, tex, str, modified tex
  """
  solnVars = obj.getSolutionExportVariableNames()
  if not solnVars:
    return tex
  msg = r"""\vspace{7pt} \\When used as part of a \xmlNode{MultiRun} step, this entity provides
        additional information through the \xmlNode{SolutionExport} DataObject. The
        following variables can be requested within the \xmlNode{SolutionExport}:
        \begin{itemize}
        """
  for var, desc in solnVars.items():
    var = var.replace('_', '\_')
    var = var.replace('{', '\{')
    var = var.replace('}', '\}')
    msg += r"""  \item \texttt{""" + var + r"""}: """ + desc + r"""
           """
  msg += r"""
         \end{itemize}"""

  split = tex.split('\n')
  for l, line in enumerate(split):
    if 'node recognizes the following parameters' in line:
      split.insert(l-1, msg)
      break
  tex = '\n'.join(split)
  return tex

# examples
minimalGradientDescent = r"""
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

minimalSimulatedAnnealing = r"""
\hspace{24pt}
Simulated Annealing Example:
\begin{lstlisting}[style=XML]
  <Optimizers>
    ...
    <SimulatedAnnealing name="simOpt">
      <samplerInit>
        <limit>2000</limit>
        <initialSeed>42</initialSeed>
        <writeSteps>every</writeSteps>
        <type>min</type>
      </samplerInit>
      <convergence>
        <objective>1e-6</objective>
        <temperature>1e-20</temperature>
        <persistence>1</persistence>
      </convergence>
      <coolingSchedule>
        <exponential>
          <alpha>0.94</alpha>
        </exponential>
      </coolingSchedule>
      <variable name="x">
        <distribution>beale_dist</distribution>
        <initial>-2.5</initial>
      </variable>
      <variable name="y">
        <distribution>beale_dist</distribution>
        <initial>3.5</initial>
      </variable>
      <objective>ans</objective>
      <TargetEvaluation class="DataObjects" type="PointSet">optOut</TargetEvaluation>
    </SimulatedAnnealing>
    ...
  </Optimizers>
\end{lstlisting}

"""
minimalGeneticAlgorithm = r"""
\hspace{24pt}
Genetic Algorithm Example:
\begin{lstlisting}[style=XML]
  <Optimizers>
    ...
    <GeneticAlgorithm name="GAopt">
      <samplerInit>
        <limit>50</limit>
        <initialSeed>42</initialSeed>
        <writeSteps>every</writeSteps>
      </samplerInit>

      <GAparams>
        <populationSize>20</populationSize>
        <parentSelection>rouletteWheel</parentSelection>
        <reproduction>
          <crossover type="onePointCrossover">
            <points>3</points>
            <crossoverProb>0.8</crossoverProb>
          </crossover>
          <mutation type="swapMutator">
            <locs>2,5</locs>
            <mutationProb>0.9</mutationProb>
          </mutation>
        </reproduction>
        <fitness type="invLinear">
          <a>2.0</a>
          <b>1.0</b>
        </fitness>
        <survivorSelection>fitnessBased</survivorSelection>
      </GAparams>

      <convergence>
        <objective>56</objective>
      </convergence>

      <variable name="x1">
        <distribution>uniform_dist_woRepl_1</distribution>
        <initial>1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20</initial>
      </variable>

      <variable name="x2">
        <distribution>uniform_dist_woRepl_1</distribution>
        <initial>2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1</initial>
      </variable>

      <variable name="x3">
        <distribution>uniform_dist_woRepl_1</distribution>
        <initial>3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2</initial>
      </variable>

      <variable name="x4">
        <distribution>uniform_dist_woRepl_1</distribution>
        <initial>4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3</initial>
      </variable>

      <variable name="x5">
        <distribution>uniform_dist_woRepl_1</distribution>
        <initial>5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4</initial>
      </variable>

      <variable name="x6">
        <distribution>uniform_dist_woRepl_1</distribution>
        <initial>6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5</initial>
      </variable>

      <objective>ans</objective>
      <TargetEvaluation class="DataObjects" type="PointSet">optOut</TargetEvaluation>
    </GeneticAlgorithm>
    ...
  </Optimizers>
\end{lstlisting}

"""

minimalBayesianOptimizer = r"""
\hspace{24pt}
Bayesian Optimizer Example:
\begin{lstlisting}[style=XML]
  <Optimizers>
    <BayesianOptimizer name="opter">

      <objective>ans</objective>

      <variable name="x">
        <distribution>egg_dist</distribution>
      </variable>

      <variable name="y">
        <distribution>egg_dist</distribution>
      </variable>

      <TargetEvaluation class="DataObjects" type="PointSet">optOut</TargetEvaluation>

      <samplerInit>
        <limit>50</limit>
        <initialSeed>42</initialSeed>
        <writeSteps>every</writeSteps>
      </samplerInit>

      <Sampler    class="Samplers"  type="Stratified" >LHS_samp</Sampler>

      <ROM  class="Models" type="ROM">gpROM</ROM>

      <Acquisition>
        <ExpectedImprovement>
          <optimizationMethod>differentialEvolution</optimizationMethod>
          <seedingCount>30</seedingCount>
        </ExpectedImprovement>
      </Acquisition>

    </BayesianOptimizer>
  </Optimizers>
\end{lstlisting}

"""
# examples Factory
exampleFactory = {'GradientDescent':minimalGradientDescent,'SimulatedAnnealing':minimalSimulatedAnnealing,'GeneticAlgorithm':minimalGeneticAlgorithm,'BayesianOptimizer':minimalBayesianOptimizer}

#------------#
# OPTIMIZERS #
#------------#
from ravenframework import Optimizers
msg = ''
# base classes first
optDescr = wrapText(Optimizers.Optimizer.userManualDescription(), '  ')
msg += optDescr
# write all known types
for name in Optimizers.factory.knownTypes():
  obj = Optimizers.factory.returnClass(name)
  specs = obj.getInputSpecification()
  tex = specs.generateLatex()
  tex = insertSolnExport(tex, obj)
  msg +=tex
  msg+= exampleFactory[name]

fName = os.path.abspath(os.path.join(os.path.dirname(__file__), 'optimizer.tex'))
with open(fName, 'w') as f:
  f.writelines(msg)

print(f'\nSuccessfully wrote "{fName}"')
