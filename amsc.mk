mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
MY_DIR :=  $(patsubst %/,%,$(dir $(mkfile_path)))

AMSC_srcfiles := $(shell find $(MY_DIR)/src/contrib -name "*.cpp" -not -name main.C)
AMSC_MODULE = $(MY_DIR)/src/contrib/_amsc.so

ifeq ($(NO_CONDA),1)
ifeq ($(CROW_USE_PYTHON3),TRUE)
amsc :: $(AMSC_MODULE)
$(AMSC_MODULE) : $(MY_DIR)/src/contrib/amsc.i $(AMSC_srcfiles)
	@echo "Building "$@"..."
	(cd $(MY_DIR) && unset CXX CC && if test `uname` = Darwin; then MACOSX_DEPLOYMENT_TARGET=10.12; export MACOSX\
_DEPLOYMENT_TARGET; fi && python3 ./setup3.py build_ext build install --install-platlib=./framework/contrib/AMSC)
	@echo "Done"
else
amsc :: $(AMSC_MODULE)
$(AMSC_MODULE) : $(MY_DIR)/src/contrib/amsc.i $(AMSC_srcfiles)
	@echo "Building "$@"..."
	(cd $(MY_DIR) && unset CXX CC && if test `uname` = Darwin; then MACOSX_DEPLOYMENT_TARGET=10.12; export MACOSX\
_DEPLOYMENT_TARGET; fi && python ./setup.py build_ext build install --install-platlib=./framework/contrib/AMSC)
	@echo "Done"
endif
else
ifeq ($(CROW_USE_PYTHON3),TRUE)
amsc :: $(AMSC_MODULE)
$(AMSC_MODULE) : $(MY_DIR)/src/contrib/amsc.i $(AMSC_srcfiles)
	@echo "Building "$@"..."
	(cd $(MY_DIR) && unset CXX CC && if test `uname` = Darwin; then MACOSX_DEPLOYMENT_TARGET=10.12; export MACOSX\
_DEPLOYMENT_TARGET; fi && . $(MY_DIR)/scripts/establish_conda_env.sh --load && python3 ./setup3.py build_ext build install --install-platlib=./framework/contrib/AMSC)
	@echo "Done"
else
amsc :: $(AMSC_MODULE)
$(AMSC_MODULE) : $(MY_DIR)/src/contrib/amsc.i $(AMSC_srcfiles)
	@echo "Building "$@"..."
	(cd $(MY_DIR) && unset CXX CC && if test `uname` = Darwin; then MACOSX_DEPLOYMENT_TARGET=10.12; export MACOSX\
_DEPLOYMENT_TARGET; fi && . $(MY_DIR)/scripts/establish_conda_env.sh --load && python ./setup.py build_ext build install --install-platlib=./framework/contrib/AMSC)
	@echo "Done"
endif
endif
