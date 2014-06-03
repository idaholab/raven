RAVEN_DIR := $(ROOT_DIR)/raven

PYTHON3_HELLO := $(shell python3 -c "print('HELLO')" 2>/dev/null)
PYTHON2_HELLO := $(shell python -c "print 'HELLO'" 2>/dev/null)

SWIG_VERSION := $(shell swig -version 2>/dev/null)
PYTHON_CONFIG_WHICH := $(shell which python-config 2>/dev/null)

UNAME := $(shell uname)

ifneq ($(PYTHON_CONFIG_WHICH),)
	PYTHON2_INCLUDE=$(shell python-config --includes)
	PYTHON2_LIB=$(shell python-config --ldflags)
endif

# look for numpy include directory
#NUMPY_INCLUDE = $(shell python $(RAVEN_DIR)/scripts/find_numpy_include.py)

ifeq ($(PYTHON3_HELLO),HELLO)
        PYTHON_INCLUDE = $(shell $(RAVEN_DIR)/scripts/find_flags.py include) #-DPy_LIMITED_API 
        PYTHON_LIB = $(shell $(RAVEN_DIR)/scripts/find_flags.py library) #-DPy_LIMITED_API 
ifeq ($(findstring SWIG Version 2,$(SWIG_VERSION)),)
else
	SWIG_PY_FLAGS=-py3
endif #Have SWIG

else #no Python3
ifeq ($(PYTHON2_HELLO),HELLO)
ifeq ($(PYTHON_CONFIG_WHICH),)
	PYTHON_INCLUDE = -DNO_PYTHON_FOR_YOU
	PYTHON_LIB = -DNO_PYTHON_FOR_YOU
else #Python 2 and Python config found but not Python 3 
	PYTHON_INCLUDE=$(PYTHON2_INCLUDE)
	PYTHON_LIB=$(PYTHON2_LIB)
	SWIG_PY_FLAGS=
endif
else
#Python3 and Python2 not found.
	PYTHON_INCLUDE = -DNO_PYTHON_FOR_YOU
	PYTHON_LIB = -DNO_PYTHON_FOR_YOU
endif
endif

RAVEN_LIB_INCLUDE_DIR := $(HERD_TRUNK_DIR)/crow/contrib/include

ifeq  ($(UNAME),Darwin)
raven_shared_ext := dylib
else
raven_shared_ext := so
endif

HAS_DYNAMIC := $(shell $(libmesh_LIBTOOL) --config | grep build_libtool_libs | cut -d'=' -f2 )

