mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
MY_DIR :=  $(patsubst %/,%,$(dir $(mkfile_path)))

AMSC_srcfiles := $(shell find $(MY_DIR)/src/contrib -name "*.cpp" -not -name main.C)
AMSC_MODULE = $(MY_DIR)/src/contrib/_amsc.so

amsc :: $(AMSC_MODULE)
$(AMSC_MODULE) : $(MY_DIR)/src/contrib/amsc.i $(AMSC_srcfiles)
	@echo "Building "$@"..."
	(cd $(MY_DIR) && if test `uname` != "Darwin"; then unset CXX; fi && python $(MY_DIR)/setup.py build_ext build install --install-platlib=$(MY_DIR)/framework/contrib)
	@echo "Done"
