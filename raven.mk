RAVEN_DIR := $(CURR_DIR)

RAVEN_INC_DIRS := $(shell find $(RAVEN_DIR)/include -type d -not -path "*/.svn*")
RAVEN_INCLUDE  := $(foreach i, $(RAVEN_INC_DIRS), -I$(i))

libmesh_INCLUDE := $(RAVEN_INCLUDE) $(libmesh_INCLUDE)

RAVEN_LIB := $(RAVEN_DIR)/libRAVEN-$(METHOD).la

RAVEN_APP := $(RAVEN_DIR)/RAVEN-$(METHOD)

# source files
RAVEN_srcfiles    := $(shell find $(RAVEN_DIR)/src -name "*.C" -not -name main.C)
RAVEN_csrcfiles   := $(shell find $(RAVEN_DIR)/src -name "*.c")
RAVEN_fsrcfiles   := $(shell find $(RAVEN_DIR)/src -name "*.f")
RAVEN_f90srcfiles := $(shell find $(RAVEN_DIR)/src -name "*.f90")

# object files
RAVEN_objects := $(patsubst %.C, %.$(obj-suffix), $(RAVEN_srcfiles))
RAVEN_objects += $(patsubst %.c, %.$(obj-suffix), $(RAVEN_csrcfiles))
RAVEN_objects += $(patsubst %.f, %.$(obj-suffix), $(RAVEN_fsrcfiles))
RAVEN_objects += $(patsubst %.f90, %.$(obj-suffix), $(RAVEN_f90srcfiles))

# plugin files
RAVEN_plugfiles    := $(shell find $(RAVEN_DIR)/plugins/ -name "*.C" 2>/dev/null)
RAVEN_cplugfiles   := $(shell find $(RAVEN_DIR)/plugins/ -name "*.c" 2>/dev/null)
RAVEN_fplugfiles   := $(shell find $(RAVEN_DIR)/plugins/ -name "*.f" 2>/dev/null)
RAVEN_f90plugfiles := $(shell find $(RAVEN_DIR)/plugins/ -name "*.f90" 2>/dev/null)

# plugins
RAVEN_plugins := $(patsubst %.C, %-$(METHOD).plugin, $(RAVEN_plugfiles))
RAVEN_plugins += $(patsubst %.c, %-$(METHOD).plugin, $(RAVEN_cplugfiles))
RAVEN_plugins += $(patsubst %.f, %-$(METHOD).plugin, $(RAVEN_fplugfiles))
RAVEN_plugins += $(patsubst %.f90, %-$(METHOD).plugin, $(RAVEN_f90plugfiles))

# RAVEN main
RAVEN_main_src    := $(RAVEN_DIR)/src/main.C
RAVEN_app_objects := $(patsubst %.C, %.$(obj-suffix), $(RAVEN_main_src))

# dependency files
RAVEN_deps := $(patsubst %.C, %.$(obj-suffix).d, $(RAVEN_srcfiles)) \
              $(patsubst %.c, %.$(obj-suffix).d, $(RAVEN_csrcfiles)) \
              $(patsubst %.C, %.$(obj-suffix).d, $(RAVEN_main_src))

# clang static analyzer files
RAVEN_analyzer := $(patsubst %.C, %.plist.$(obj-suffix), $(RAVEN_srcfiles))

# If building shared libs, make the plugins a dependency, otherwise don't.
ifeq ($(libmesh_shared),yes)
  RAVEN_plugin_deps := $(RAVEN_plugins)
else
  RAVEN_plugin_deps :=
endif

all:: $(RAVEN_LIB) amsc

$(RAVEN_LIB): $(RAVEN_objects) $(RAVEN_plugin_deps)
	@echo "Linking "$@"..."
	@$(libmesh_LIBTOOL) --tag=CXX $(LIBTOOLFLAGS) --mode=link --quiet \
	  $(libmesh_CXX) $(libmesh_CXXFLAGS) -o $@ $(RAVEN_objects) $(libmesh_LIBS) $(libmesh_LDFLAGS) $(EXTERNAL_FLAGS) -rpath $(RAVEN_DIR)
	@$(libmesh_LIBTOOL) --mode=install --quiet install -c $(RAVEN_LIB) $(RAVEN_DIR)

# Clang static analyzer
sa:: $(RAVEN_analyzer)

################################################################################
## Swig for Approximate Morse-Smale Complex (AMSC)

AMSC_srcfiles := $(shell find $(RAVEN_DIR)/src/contrib -name "*.cpp" -not -name main.C)
amsc:: $(RAVEN_DIR)/src/contrib/amsc.i $(AMSC_srcfiles)
	@echo "Building "$@"..."
	(cd $(RAVEN_DIR) && python $(RAVEN_DIR)/setup.py build_ext build install --install-platlib=$(RAVEN_DIR)/src/contrib)
	@echo "Done"
#	swig -c++ -python $(SWIG_PY_FLAGS)  -I$(RAVEN_DIR)/include/contrib/ $(RAVEN_DIR)/src/contrib/amsc.i
#	$(CXX) -fPIC -shared $(RAVEN_DIR)/src/contrib/amsc_wrap.cxx -I$(RAVEN_DIR)/include/contrib -I/usr/include/python2.7 $(AMSC_srcfiles) -lpython2.7 -o $(RAVEN_DIR)/src/contrib/_amsc.so
################################################################################

# include RAVEN dep files
-include $(RAVEN_deps)

# how to build RAVEN application
ifeq ($(APPLICATION_NAME),RAVEN)
all:: RAVEN amsc

RAVEN: $(RAVEN_APP) $(CONTROL_MODULES) $(CROW_MODULES)

$(RAVEN_APP): $(moose_LIB) $(elk_MODULES) $(RAVEN_LIB) $(RAVEN_app_objects) $(CROW_LIB) $(app_LIBS)
	@echo "Linking "$@"..."
	$(libmesh_LIBTOOL) --tag=CXX $(LIBTOOLFLAGS) --mode=link --quiet \
          $(libmesh_CXX) $(libmesh_CXXFLAGS) -o $@ $(RAVEN_app_objects) $(RAVEN_LIB) $(elk_MODULES) $(moose_LIB) $(libmesh_LIBS) $(libmesh_LDFLAGS) $(ADDITIONAL_LIBS) $(app_LIBS) $(CROW_LIB) $(PYTHON_LIB)

RAVEN_MODULES = $(CROW_DIR)/control_modules

RAVEN_MODULE_COMPILE_LINE=@$(libmesh_LIBTOOL) --tag=CXX $(LIBTOOLFLAGS) --mode=compile --quiet $(libmesh_CXX) $(libmesh_CPPFLAGS) $(libmesh_CXXFLAGS)  $(app_INCLUDES) -DRAVEN_MODULES='"$(RAVEN_MODULES)"' $(libmesh_INCLUDE) -MMD -MF $@.d -MT $@ -c $< -o $@

$(APPLICATION_DIR)/src/executioners/RavenExecutioner.$(obj-suffix): $(APPLICATION_DIR)/src/executioners/RavenExecutioner.C
	@echo "Override RavenExecutioner Compile"
	$(RAVEN_MODULE_COMPILE_LINE)

endif

delete_list := $(RAVEN_APP) $(RAVEN_LIB) $(RAVEN_DIR)/libRAVEN-$(METHOD).*

clean::
	@rm -f $(RAVEN_DIR)/src/contrib/_amsc.so \
          $(RAVEN_DIR)/src/contrib/amsc_wrap.cxx \
          $(RAVEN_DIR)/src/contrib/amsc.py \
          $(RAVEN_DIR)/src/contrib/*egg-info \
          $(RAVEN_objects) \
          $(RAVEN_app_objects) \
          $(RAVEN_APP) \
          build/*/src/contrib/* \
          build/*/_amsc.so \
          $(RAVEN_plugins)
	@find $(RAVEN_DIR)/framework  -name '*.pyc' -exec rm '{}' \;

cleanall::
	make -C $(RAVEN_DIR) clean
