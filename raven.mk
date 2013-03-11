PYTHON3_HELLO := $(shell python3 -c "print('HELLO')" 2>/dev/null)
PYTHON2_HELLO := $(shell python -c "print 'HELLO'" 2>/dev/null)

SWIG_VERSION := $(shell swig -version 2>/dev/null)
PYTHON_CONFIG_WHICH := $(shell which python-config 2>/dev/null)

UNAME := $(shell uname)

ifeq ($(PYTHON3_HELLO),HELLO)
	PYTHON_INCLUDE = $(shell $(RAVEN_DIR)/scripts/find_flags.py include) #-DPy_LIMITED_API
	PYTHON_LIB = $(shell $(RAVEN_DIR)/scripts/find_flags.py library) #-DPy_LIMITED_API
ifeq ($(findstring SWIG Version 2,$(SWIG_VERSION)),)
	PYTHON_MODULES = 
else
	PYTHON_MODULES = $(RAVEN_DIR)/python_modules/_distribution1D.so $(RAVEN_DIR)/python_modules/_raventools.so
endif

else
ifeq ($(PYTHON2_HELLO),HELLO)
ifeq ($(PYTHON_CONFIG_WHICH),)
	PYTHON_INCLUDE = -DNO_PYTHON3_FOR_YOU
	PYTHON_LIB = -DNO_PYTHON3_FOR_YOU
	PYTHON_MODULES = 
else
	PYTHON_INCLUDE=$(shell python-config --includes)
	PYTHON_LIB=$(shell python-config --libs)
	PYTHON_MODULES=
#PYTHON_MODULES=$(RAVEN_DIR)/python_modules/_distribution1D.so $(RAVEN_DIR)/python_modules/_raventools.so
endif
else
#Python3 not found.
	PYTHON_INCLUDE = -DNO_PYTHON3_FOR_YOU
	PYTHON_LIB = -DNO_PYTHON3_FOR_YOU
	PYTHON_MODULES = 
endif
endif

ifeq  ($(UNAME),Darwin)
raven_shared_ext := dylib
else
raven_shared_ext := so
endif

HAS_DYNAMIC := $(shell PYTHONPATH=$(RAVEN_DIR)/../moose/tests/ python -c 'import util; print "DYNAMIC" in util.getSharedOption("../libmesh/installed")')

ifeq ($(HAS_DYNAMIC),False)  
ifdef PYTHON_MODULES
$(error RAVEN modules must be compiled with shared libmesh libraries)
endif
endif


#ifeq ($(UNAME),Darwin)
EXTRA_MOOSE_LIBS = -L$(MOOSE_DIR) -lmoose-$(METHOD) -L$(RAVEN_DIR) -lRAVEN-$(METHOD) -L$(R7_DIR) -lr7_moose-$(METHOD) -L$(ELK_DIR)/lib -lelk-$(METHOD) -lphase_field-$(METHOD) -lnavier_stokes-$(METHOD) -lheat_conduction-$(METHOD) -lsolid_mechanics-$(METHOD) -ltensor_mechanics-$(METHOD) -llinear_elasticity-$(METHOD) -lchemical_reactions-$(METHOD) -lfluid_mass_energy_balance-$(METHOD) -lmisc-$(METHOD) -lcontact-$(METHOD) $(libmesh_LIBS)
#else
#	EXTRA_MOOSE_LIBS = 
#endif


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

# If building shared libs, make the plugins a dependency, otherwise don't.
ifeq ($(libmesh_shared),yes)
  RAVEN_plugin_deps := $(RAVEN_plugins)
else
  RAVEN_plugin_deps :=
endif

all:: $(RAVEN_LIB)

$(RAVEN_LIB): $(RAVEN_objects) $(RAVEN_plugin_deps)
	@echo "Linking "$@"..."
	@$(libmesh_LIBTOOL) --tag=CXX $(LIBTOOLFLAGS) --mode=link --quiet \
	  $(libmesh_CXX) $(libmesh_CXXFLAGS) -o $@ $(RAVEN_objects) $(libmesh_LIBS) $(libmesh_LDFLAGS) $(EXTERNAL_FLAGS) -rpath $(RAVEN_DIR)
	@$(libmesh_LIBTOOL) --mode=install --quiet install -c $(RAVEN_LIB) $(RAVEN_DIR)

# include RAVEN dep files
-include $(RAVEN_deps)

# how to build RAVEN application
ifeq ($(APPLICATION_NAME),RAVEN)
all:: RAVEN

RAVEN_MODULES = $(RAVEN_DIR)/python_modules

$(RAVEN_DIR)/src/executioners/PythonControl.$(obj-suffix): $(RAVEN_DIR)/src/executioners/PythonControl.C
	@echo "Override PythonControl Compile"
	@$(libmesh_LIBTOOL) --tag=CXX $(LIBTOOLFLAGS) --mode=compile --quiet \
          $(libmesh_CXX) $(libmesh_CPPFLAGS) $(libmesh_CXXFLAGS) $(PYTHON_INCLUDE) -DRAVEN_MODULES='"$(RAVEN_MODULES)"' $(libmesh_INCLUDE) -MMD -MF $@.d -MT $@ -c $< -o $@

# TODO[JWP]: Should this use libtool to make a platform-independent shared library?
#            I could not test it because I don't have python3.
$(RAVEN_DIR)/python_modules/_distribution1D.so : $(RAVEN_DIR)/python_modules/distribution1D.i \
                                                 $(RAVEN_DIR)/src/distributions/distribution_1D.C \
                                                 $(RAVEN_DIR)/src/distributions/DistributionContainer.C \
                                                 $(RAVEN_DIR)/src/utilities/Interpolation_Functions.C $(RAVEN_LIB)
# Swig
	swig -c++ -python -py3 -I$(RAVEN_DIR)/../moose/include/base/ $(libmesh_INCLUDE) \
          -I$(RAVEN_DIR)/../moose/include/utils/ -I$(RAVEN_DIR)/include/distributions/ \
          -I$(RAVEN_DIR)/include/utilities/ -I$(RAVEN_DIR)/include/base/ \
          $(RAVEN_MODULES)/distribution1D.i
# Compile
	$(libmesh_LIBTOOL) --tag=CXX $(LIBTOOLFLAGS) --mode=compile \
	$(libmesh_CXX) $(libmeh_CPPFLAGS) $(PYTHON_INCLUDE) $(libmesh_INCLUDE) \
	 -c  $(RAVEN_MODULES)/distribution1D_wrap.cxx -o $(RAVEN_DIR)/python_modules/distribution1D_wrap.lo
	$(libmesh_LIBTOOL) --tag=CXX $(LIBTOOLFLAGS) --mode=link \
	 $(libmesh_CXX) $(libmesh_CXXFLAGS) \
	-shared -o $(RAVEN_MODULES)/libdistribution1D.la $(RAVEN_LIB) $(PYTHON_LIB) $(RAVEN_MODULES)/distribution1D_wrap.lo -rpath $(RAVEN_MODULES)
	$(libmesh_LIBTOOL) --tag=CXX $(LIBTOOLFLAGS) --mode=install install -c $(RAVEN_MODULES)/libdistribution1D.la  $(RAVEN_MODULES)/libdistribution1D.la 
	rm -f $(RAVEN_MODULES)/_distribution1D.so
	ln -s libdistribution1D.$(raven_shared_ext) $(RAVEN_MODULES)/_distribution1D.so


$(RAVEN_DIR)/python_modules/_raventools.so : $(RAVEN_DIR)/python_modules/raventools.i \
                                             $(RAVEN_DIR)/src/tools/batteries.C \
                                             $(RAVEN_DIR)/src/tools/dieselGenerator.C \
                                             $(RAVEN_DIR)/src/tools/pumpCoastdown.C \
                                             $(RAVEN_DIR)/src/tools/decayHeat.C \
                                             $(RAVEN_DIR)/src/tools/powerGrid.C \
                                             $(RAVEN_DIR)/src/utilities/Interpolation_Functions.C $(RAVEN_LIB)
# Swig
	swig -c++ -python -py3 -I$(RAVEN_DIR)/include/tools/  -I$(RAVEN_DIR)/include/utilities/ $(RAVEN_DIR)/python_modules/raventools.i
# Compile
	$(libmesh_CXX) $(libmesh_CPPFLAGS) $(libmesh_CXXFLAGS) $(PYTHON_INCLUDE) -fPIC \
          -I$(RAVEN_DIR)/include/tools/ -I$(RAVEN_DIR)/include/utilities/ \
          $(RAVEN_DIR)/python_modules/raventools_wrap.cxx \
          $(RAVEN_DIR)/src/tools/*.C \
          $(RAVEN_DIR)/src/utilities/Interpolation_Functions.C -o $(RAVEN_DIR)/python_modules/_raventools.so $(PYTHON_LIB)


RAVEN: $(RAVEN_APP) $(PYTHON_MODULES)

$(RAVEN_APP): $(moose_LIB) $(elk_MODULES) $(r7_LIB) $(RAVEN_LIB) $(RAVEN_app_objects)
	@echo "Linking "$@"..."
	@$(libmesh_LIBTOOL) --tag=CXX $(LIBTOOLFLAGS) --mode=link --quiet \
          $(libmesh_CXX) $(libmesh_CXXFLAGS) -o $@ $(RAVEN_app_objects) $(RAVEN_LIB) $(r7_LIB) $(elk_MODULES) $(moose_LIB) $(libmesh_LIBS) $(libmesh_LDFLAGS) $(ADDITIONAL_LIBS) $(PYTHON_LIB)

endif

delete_list := $(RAVEN_APP) $(RAVEN_LIB) $(RAVEN_DIR)/libRAVEN-$(METHOD).*

clean::
	@rm -f $(RAVEN_DIR)/python_modules/_distribution1D.so \
          $(RAVEN_DIR)/python_modules/_raventools.so \
          $(RAVEN_DIR)/python_modules/distribution1D_wrap.cxx \
          $(RAVEN_DIR)/python_modules/raventools_wrap.cxx \
          $(RAVEN_DIR)/python_modules/distribution1D.py \
          $(RAVEN_DIR)/python_modules/libdistribution1D.*

clobber::
	@rm -f $(RAVEN_DIR)/python_modules/_distribution1D.so \
          $(RAVEN_DIR)/python_modules/distribution1D_wrap.cxx \
          $(RAVEN_DIR)/python_modules/distribution1D.py

cleanall:: 
	make -C $(RAVEN_DIR) clean 
