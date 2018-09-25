###############################################################################
################### MOOSE Application Standard Makefile #######################
###############################################################################
#
# Optional Environment variables
# MOOSE_DIR        - Root directory of the MOOSE project
# HERD_TRUNK_DIR   - Location of the HERD repository
# FRAMEWORK_DIR    - Location of the MOOSE framework
#
###############################################################################
MOOSE_SUBMODULE    := $(CURDIR)/moose
ifneq ($(wildcard $(MOOSE_SUBMODULE)/framework/Makefile),)
  MOOSE_DIR        ?= $(MOOSE_SUBMODULE)
else
  MOOSE_DIR        ?= $(CURDIR)/../moose
endif
HERD_TRUNK_DIR     ?= $(shell dirname `pwd`)
FRAMEWORK_DIR      ?= $(MOOSE_DIR)/framework
###############################################################################
CURR_DIR    := $(CURDIR)

# touch hit.cpp to make sure its time stamp is different than hit.pyx
## this is not a clean solution, but hopefully it prevents asking to use cython
CYTHON_AVOIDANCE_ACTION=$(shell touch $(MOOSE_DIR)/framework/contrib/hit/hit.cpp)


# framework
#include $(FRAMEWORK_DIR)/build.mk
#include $(FRAMEWORK_DIR)/moose.mk

################################## MODULES ####################################
#HEAT_CONDUCTION   := yes
#MISC              := yes
#FLUID_PROPERTIES  := yes
#include           $(MOOSE_DIR)/modules/modules.mk
###############################################################################
# CROW
CROW_SUBMODULE     := $(CURR_DIR)/crow
ifneq ($(wildcard $(CROW_SUBMODULE)/Makefile),)
  CROW_DIR         ?= $(CROW_SUBMODULE)
else
  CROW_DIR         ?= $(HERD_TRUNK_DIR)/crow
endif

all::

APPLICATION_DIR    := $(CROW_DIR)
APPLICATION_NAME   := CROW
include 	   $(CROW_DIR)/config.mk
include            $(CROW_DIR)/crow.mk
include            $(CROW_DIR)/crow_python_modules.mk

# RAVEN
APPLICATION_DIR    := $(CURR_DIR)
APPLICATION_NAME   := RAVEN
include $(CURR_DIR)/raven.mk

###############################################################################
# Additional special case targets should be added here

