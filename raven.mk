RAVEN_DIR := $(CURR_DIR)
#Conda doesn't work with anything but bash and zsh
SHELL := /bin/bash

################################################################################
## Build system for Approximate Morse-Smale Complex (AMSC)
include $(RAVEN_DIR)/amsc.mk
###############################################################################

################################################################################
## Build system for "hit", required by moose regression test system
hit $(MOOSE_DIR)/python/hit.so:: $(FRAMEWORK_DIR)/contrib/hit/hit.cpp $(FRAMEWORK_DIR)/contrib/hit/lex.cc $(FRAMEWORK_DIR)/contrib/hit/parse.cc
	bash -c 'cd scripts/TestHarness/hit-windows && ./build_hit.sh'
###############################################################################

framework_modules:: amsc python_crow_modules hit

all:: amsc python_crow_modules hit

####################################################################################
#           find and remove all the *.pyc files (better safe then sorry)           #
$(shell find $(RAVEN_DIR)/framework -type f -name "*.pyc" -exec rm {} +)           #
####################################################################################

delete_list := $(RAVEN_APP) $(RAVEN_LIB) $(RAVEN_DIR)/libRAVEN-$(METHOD).*

clean::
	@rm -f $(RAVEN_DIR)/framework/contrib/AMSC/_amsc.so \
          $(RAVEN_DIR)/framework/contrib/AMSC/amsc*egg-info \
          $(RAVEN_DIR)/framework/contrib/AMSC/amsc.py \
          $(RAVEN_DIR)/framework/contrib/AMSC/amsc.pyc \
          $(RAVEN_DIR)/src/contrib/amsc_wrap.cxx \
          $(RAVEN_DIR)/src/contrib/amsc_wrap.cpp \
          $(RAVEN_DIR)/src/contrib/amsc.py \
          $(RAVEN_DIR)/src/contrib/amsc.pyc \
          $(RAVEN_objects) \
          $(RAVEN_app_objects) \
          $(RAVEN_APP) \
          $(RAVEN_plugins) \
	  $(MOOSE_DIR)/python/hit.so \
	  $(MOOSE_DIR)/python/hit.pyd
	@rm -Rf $(RAVEN_DIR)/build $(FRAMEWORK_DIR)/contrib/hit/build
	@find $(RAVEN_DIR)/framework  -name '*.pyc' -exec rm '{}' \;
	$(MAKE) -C $(FRAMEWORK_DIR)/contrib/hit clean

cleanall::
	make -C $(RAVEN_DIR) clean
