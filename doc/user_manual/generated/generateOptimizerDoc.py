# TODO HEADER
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

#examples
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
# examples
exampleFactory = {'GradientDescent':minimalGradientDescent,'SimulatedAnnealing':minimalSimulatedAnnealing}

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
  all += exampleFactory[name]

fName = os.path.abspath(os.path.join(os.path.dirname(__file__), 'optimizer.tex'))
with open(fName, 'w') as f:
  f.writelines(all)
