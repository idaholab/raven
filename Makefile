
CURR_DIR    := $(CURDIR)

# CROW
CROW_SUBMODULE     := $(CURR_DIR)/crow
ifneq ($(wildcard $(CROW_SUBMODULE)/Makefile),)
  CROW_DIR         ?= $(CROW_SUBMODULE)
else
  $(warning CROW_DIR not found)
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

