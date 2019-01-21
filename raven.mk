RAVEN_DIR := $(CURR_DIR)
#Conda doesn't work with anything but bash and zsh
SHELL := /bin/bash

################################################################################
## Build system for Approximate Morse-Smale Complex (AMSC)
include $(RAVEN_DIR)/amsc.mk
###############################################################################

framework_modules:: amsc python_crow_modules

all:: amsc python_crow_modules

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

cleanall::
	make -C $(RAVEN_DIR) clean
