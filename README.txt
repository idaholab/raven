RAVEN

Based upon the MOOSE HPC framework, RAVEN (Reactor Analysis and
Virtual Control Environment) is a multi-tasking application focused on
RELAP-7 simulation control, reactor plant control logic, reactor
system analysis, uncertainty quantification, and performing
probability risk assessments (PRA) for postulated events.

Raven has been worked on by at least:
Alfonsi, Andrea (alfoa)
Andrs, David (andrsd)
Kinoshita, Robert A (bobk)
Cogliati, Joshua J (cogljj)
Rabiti, Cristian (crisr)
Gaston, Derek R (gastdr)
Mandelli, Diego (mandd)
Miller, Jason M (milljm)
Nielsen, Joseph W (nieljw)
Permann, Cody J (permcj)
Peterson, JW (petejw)
Schoen, Scott A (schosa)
Swiler, Laura P (swillp)
Talbot, Paul W (talbpw)
Zhao, Haihua (zhaoh)

A list can be gotten by:
svn log -q | awk '/^r/ {print $3}' | sort | uniq -c

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
