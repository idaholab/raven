RAVEN

RAVEN is a generic software framework designed to perform parametric and stochastic analysis
based on the response of complex system codes. The initial development was aimed to provide
dynamic risk analysis capabilities to the Thermo-Hydraulic code RELAP-7, currently under
development at the Idaho National Laboratory (INL).
Although the initial goal has been fully accomplished, RAVEN is now a multi-purpose probabilistic
and uncertainty quantification platform, capable to communicate with any system code. This agnosticism
includes providing Application Programming Interfaces (APIs). These APIs allow RAVEN to interact with
any code as long as all the parameters that need to be perturbed are accessible by inputs files or via
python interfaces. RAVEN is capable of investigating the system response, and investigating the input
space using Monte Carlo, Grid, or Latin Hyper Cube sampling schemes, but its strength is focused
toward system feature discovery, such as limit surfaces, separating regions of the input space leading
to system failure,  using dynamic supervised learning techniques. The development of RAVEN has started
in 2012, when, within the Nuclear Energy Advanced Modeling and Simulation (NEAMS) program, the need to
provide a modern risk evaluation framework became stronger. RAVEN principal assignment is to provide the
necessary software and algorithms in order to employ the concept developed by the Risk Informed Safety
Margin Characterization (RISMC) program. RISMC is one of the pathways defined within the Light Water
Reactor Sustainability (LWRS) program. In the RISMC approach, the goal is not just the individuation of
the frequency of an event potentially leading to a system failure, but the closeness (or not) to key
safety-related events. Hence, the approach is interested in identifying and increasing the safety margins
related to those events. A safety margin is a numerical value quantifying the probability that a safety
metric (e.g. for an important process such as peak pressure in a pipe) is exceeded under certain conditions.
The initial development of RAVEN has been focused on providing dynamic risk assessment capability to
RELAP-7, currently under development at the INL and, likely, future replacement of the RELAP5-3D code.
Most the capabilities that have been implemented having RELAP-7 as principal focus are easily deployable
for other system codes.

Principal Investigator: Rabiti, Cristian (crisr)

Raven copyright holders are:
Alfonsi, Andrea (alfoa)
Rabiti, Cristian (crisr)
Mandelli, Diego (mandd)
Cogliati, Joshua J (cogljj)
Kinoshita, Robert A (bobk)

Past developers:
Sen, Ramazan S (senrs)


A list of other contributors can be obtained with the following command:
```git shortlog -sn```

See the file LICENSE for copyright and export information.
