PYTHON3_HELLO := $(shell python3 -c "print('HELLO')" 2>/dev/null)
PYTHON2_HELLO := $(shell python -c "print 'HELLO'" 2>/dev/null)

SWIG_VERSION := $(shell swig -version 2>/dev/null)
PYTHON_CONFIG_WHICH := $(shell which python-config 2>/dev/null)

UNAME := $(shell uname)

ifneq ($(PYTHON_CONFIG_WHICH),)
	PYTHON2_INCLUDE=$(shell python-config --includes)
	PYTHON2_LIB=$(shell python-config --ldflags)
endif

ifeq ($(PYTHON3_HELLO),HELLO)
        PYTHON_INCLUDE = $(shell $(RAVEN_DIR)/scripts/find_flags.py include) #-DPy_LIMITED_API 
        PYTHON_LIB = $(shell $(RAVEN_DIR)/scripts/find_flags.py library) #-DPy_LIMITED_API 
ifeq ($(findstring SWIG Version 2,$(SWIG_VERSION)),)
	CONTROL_MODULES = 
	PYTHON_MODULES = 
else
	CONTROL_MODULES = $(RAVEN_DIR)/control_modules/_distribution1D.so $(RAVEN_DIR)/control_modules/_raventools.so 
	PYTHON_MODULES = $(RAVEN_DIR)/python_modules/_distribution1Dpy2.so $(RAVEN_DIR)/python_modules/_distribution1Dpy3.so $(RAVEN_DIR)/python_modules/_interpolationNDpy2.so $(RAVEN_DIR)/python_modules/_interpolationNDpy3.so
endif

else
ifeq ($(PYTHON2_HELLO),HELLO)
ifeq ($(PYTHON_CONFIG_WHICH),)
	PYTHON_INCLUDE = -DNO_PYTHON3_FOR_YOU
	PYTHON_LIB = -DNO_PYTHON3_FOR_YOU
	CONTROL_MODULES = 
	PYTHON_MODULES = 
else
	PYTHON_INCLUDE=$(PYTHON2_INCLUDE)
	PYTHON_LIB=$(PYTHON2_LIB)
	CONTROL_MODULES=
	PYTHON_MODULES = 
#CONTROL_MODULES=$(RAVEN_DIR)/control_modules/_distribution1D.so $(RAVEN_DIR)/control_modules/_raventools.so
endif
else
#Python3 not found.
	PYTHON_INCLUDE = -DNO_PYTHON3_FOR_YOU
	PYTHON_LIB = -DNO_PYTHON3_FOR_YOU
	CONTROL_MODULES = 
	PYTHON_MODULES = $(RAVEN_DIR)/python_modules/_distribution1Dpy2.so $(RAVEN_DIR)/python_modules/_interpolationNDpy2.so
endif
endif

RAVEN_LIB_INCLUDE_DIR := $(RAVEN_DIR)/contrib/include

ifeq  ($(UNAME),Darwin)
raven_shared_ext := dylib
else
raven_shared_ext := so
endif

HAS_DYNAMIC := $(shell $(libmesh_LIBTOOL) --config | grep build_libtool_libs | cut -d'=' -f2 )

ifeq ($(HAS_DYNAMIC),no)  
ifdef CONTROL_MODULES
$(warning RAVEN modules must be compiled with shared libmesh libraries)
	CONTROL_MODULES = 
endif
endif
