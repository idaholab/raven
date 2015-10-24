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
related to those events. A safety margin is a numerical value quantifying the probability that a safety metric 
(e.g. for an important process such as peak pressure in a pipe) is exceeded under certain conditions. 
The initial development of RAVEN has been focused on providing dynamic risk assessment capability to 
RELAP-7, currently under development at the INL and, likely, future replacement of the RELAP5-3D code. 
Most the capabilities that have been implemented having RELAP-7 as principal focus are easily deployable 
for other system codes.

Raven developers are:
Alfonsi, Andrea (alfoa)
Rabiti, Cristian (crisr)
Mandelli, Diego (mandd)
Cogliati, Joshua J (cogljj)
Sen, Ramazan S (senrs)
Wang, Congjian (wangc)
Maljovec, Daniel P (maljdan)
Talbot, Paul W (talbpw)
Kinoshita, Robert A (bobk)

A list of other contributors can be obtained with the following command:
```git shortlog -sn```

Directories:

control_modules - These are modules to be used by the control programs

developer_tools - Extra tools for developers (such as a relap7 to raven input converter tool)

doc - doxygen directories

framework - The directories for the RAVEN framework that allows multiple RAVEN's to be run, branched etc.

gui - Code for the graphical interface peacock

include - C++ include files

inputs - Extra example inputs

papers - Papers about RAVEN

scripts - This is the location of the test harness and compiling scripts.

src - C++ compilation files

tests - The tests for the test harness (run by run_tests)

work_in_progress - Extra code or examples that might be useful in the future.


COPYING:

NOTICE: This computer software, “Risk Analysis Virtual ENvironment
(RAVEN)”, was prepared by Battelle Energy Alliance, LLC, hereinafter
the Contractor, under Contract No. DE-AC07-05ID14517 with the United
States (U.S.) Department of Energy (DOE).  For ten years from April 2,
2014, the Government is granted for itself and others acting on its
behalf a nonexclusive, paid-up, irrevocable worldwide license in this
data to reproduce, prepare derivative works, and perform publicly and
display publicly, by or on behalf of the Government. There is
provision for the possible extension of the term of this license.
Subsequent to that period or any extension granted, the Government is
granted for itself and others acting on its behalf a nonexclusive,
paid-up, irrevocable worldwide license in this data to reproduce,
prepare derivative works, distribute copies to the public, perform
publicly and display publicly, and to permit others to do so.  The
specific term of the license can be identified by inquiry made to
Contractor or DOE.  NEITHER THE UNITED STATES NOR THE UNITED STATES
DEPARTMENT OF ENERGY, NOR CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR
IMPLIED, OR ASSUMES ANY LIABILITY OR RESPONSIBILITY FOR THE USE,
ACCURACY, COMPLETENESS, OR USEFULNESS OR ANY INFORMATION, APPARATUS,
PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
INFRINGE PRIVATELY OWNED RIGHTS.

EXPORT RESTRICTIONS: The provider of this computer software and its
employees and its agents are subject to U.S. export control laws that
prohibit or restrict (i) transactions with certain parties, and (ii)
the type and level of technologies and services that may be exported.
You agree to comply fully with all laws and regulations of the United
States and other countries (Export Laws) to assure that neither this
computer software, nor any direct products thereof are (1) exported,
directly or indirectly, in violation of Export Laws, or (2) are used
for any purpose prohibited by Export Laws, including, without
limitation, nuclear, chemical, or biological weapons proliferation.

None of this computer software or underlying information or technology
may be downloaded or otherwise exported or re-exported (i) into (or to
a national or resident of) Cuba, North Korea, Iran, Sudan, Syria or
any other country to which the U.S. has embargoed goods; or (ii) to
anyone on the U.S. Treasury Department's List of Specially Designated
Nationals or the U.S. Commerce Department's Denied Persons List,
Unverified List, Entity List, Nonproliferation Sanctions or General
Orders.  By downloading or using this computer software, you are
agreeing to the foregoing and you are representing and warranting that
you are not located in, under the control of, or a national or
resident of any such country or on any such list, and that you
acknowledge you are responsible to obtain any necessary
U.S. government authorization to ensure compliance with U.S. law.