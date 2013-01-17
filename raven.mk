RAVEN_SRC_DIRS := $(RAVEN_DIR)/src/*/*

PYTHON3_HELLO = $(shell python3 -c "print('HELLO')")
PYTHON2_HELLO = $(shell python -c "print 'HELLO'")

SWIG_VERSION = $(shell swig -version)

UNAME = $(shell uname)

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
	PYTHON_INCLUDE=$(shell python2-config --includes)
	PYTHON_LIB=$(shell python2-config --libs)
	PYTHON_MODULES=
#PYTHON_MODULES=$(RAVEN_DIR)/python_modules/_distribution1D.so $(RAVEN_DIR)/python_modules/_raventools.so
else
#Python3 not found.
	PYTHON_INCLUDE = -DNO_PYTHON3_FOR_YOU
	PYTHON_LIB = -DNO_PYTHON3_FOR_YOU
	PYTHON_MODULES = 
endif
endif



ifeq ($(UNAME),Darwin)
	EXTRA_MOOSE_LIBS = $(moose_LIB) $(libmesh_LIBS)
else
	EXTRA_MOOSE_LIBS = 
endif


RAVEN_INC_DIRS := $(shell find $(RAVEN_DIR)/include -type d -not -path "*/.svn*")
RAVEN_INCLUDE  := $(foreach i, $(RAVEN_INC_DIRS), -I$(i))

libmesh_INCLUDE := $(RAVEN_INCLUDE) $(libmesh_INCLUDE)

RAVEN_LIB := $(RAVEN_DIR)/libRAVEN-$(METHOD)$(libext)

RAVEN_APP := $(RAVEN_DIR)/RAVEN-$(METHOD)

# source files
RAVEN_srcfiles    := $(shell find $(RAVEN_SRC_DIRS) -name *.C)
RAVEN_csrcfiles   := $(shell find $(RAVEN_SRC_DIRS) -name *.c)
RAVEN_fsrcfiles   := $(shell find $(RAVEN_SRC_DIRS) -name *.f)
RAVEN_f90srcfiles := $(shell find $(RAVEN_SRC_DIRS) -name *.f90)
# object files
RAVEN_objects	:= $(patsubst %.C, %.$(obj-suffix), $(RAVEN_srcfiles))
RAVEN_objects	+= $(patsubst %.c, %.$(obj-suffix), $(RAVEN_csrcfiles))
RAVEN_objects += $(patsubst %.f, %.$(obj-suffix), $(RAVEN_fsrcfiles))
RAVEN_objects += $(patsubst %.f90, %.$(obj-suffix), $(RAVEN_f90srcfiles))

RAVEN_app_objects := $(patsubst %.C, %.$(obj-suffix), $(RAVEN_DIR)/src/main.C)

# plugin files
RAVEN_plugfiles   := $(shell find $(RAVEN_DIR)/plugins/ -name *.C 2>/dev/null)
RAVEN_cplugfiles  := $(shell find $(RAVEN_DIR)/plugins/ -name *.c 2>/dev/null)
RAVEN_fplugfiles  := $(shell find $(RAVEN_DIR)/plugins/ -name *.f 2>/dev/null)
RAVEN_f90plugfiles:= $(shell find $(RAVEN_DIR)/plugins/ -name *.f90 2>/dev/null)

# plugins
RAVEN_plugins     := $(patsubst %.C, %-$(METHOD).plugin, $(RAVEN_plugfiles))
RAVEN_plugins     += $(patsubst %.c, %-$(METHOD).plugin, $(RAVEN_cplugfiles))
RAVEN_plugins     += $(patsubst %.f, %-$(METHOD).plugin, $(RAVEN_fplugfiles))
RAVEN_plugins     += $(patsubst %.f90, %-$(METHOD).plugin, $(RAVEN_f90plugfiles))

all:: $(RAVEN_LIB)

# build rule for lib RAVEN
ifeq ($(enable-shared),yes)
# Build dynamic library
$(RAVEN_LIB): $(RAVEN_objects) $(RAVEN_plugins)
	@echo "Linking "$@"..."
	@$(libmesh_CC) $(libmesh_CXXSHAREDFLAG) -o $@ $(RAVEN_objects) $(libmesh_LDFLAGS)
else
# Build static library
ifeq ($(findstring darwin,$(hostos)),darwin)
$(RAVEN_LIB): $(RAVEN_objects)
	@echo "Linking "$@"..."
	@libtool -static -o $@ $(RAVEN_objects)
else
$(RAVEN_LIB): $(RAVEN_objects)
	@echo "Linking "$@"..."
	@$(AR) rv $@ $(RAVEN_objects)
endif
endif

# include RAVEN dep files
-include $(RAVEN_DIR)/src/*/*.d


# how to build RAVEN application
ifeq ($(APPLICATION_NAME),RAVEN)
all:: RAVEN

RAVEN_MODULES = $(RAVEN_DIR)/python_modules

$(RAVEN_DIR)/src/executioners/PythonControl.$(obj-suffix): $(RAVEN_DIR)/src/executioners/PythonControl.C
	@echo "Override PythonControl Compile"
	$(libmesh_CXX) $(libmesh_CPPFLAGS) $(libmesh_CXXFLAGS) $(PYTHON_INCLUDE) -DRAVEN_MODULES='"$(RAVEN_MODULES)"' -MMD -MF $@.d $(libmesh_INCLUDE) -c $< -o $@ 


$(RAVEN_DIR)/python_modules/_distribution1D.so : $(RAVEN_DIR)/python_modules/distribution1D.i  $(RAVEN_DIR)/src/distributions/distribution_1D.C $(RAVEN_DIR)/src/distributions/DistributionContainer.C $(RAVEN_DIR)/src/utilities/Interpolation_Functions.C
	swig -c++ -python -py3 -I$(RAVEN_DIR)/../moose/include/base/ $(libmesh_INCLUDE) -I$(RAVEN_DIR)/../moose/include/utils/ -I$(RAVEN_DIR)/include/distributions/ -I$(RAVEN_DIR)/include/utilities/ -I$(RAVEN_DIR)/include/base/ $(RAVEN_DIR)/python_modules/distribution1D.i
	$(libmesh_CXX) $(libmesh_CPPFLAGS) $(libmesh_CXXFLAGS) $(PYTHON_INCLUDE) -fPIC $(libmesh_INCLUDE) -I$(RAVEN_DIR)/include/distributions/ -I$(RAVEN_DIR)/include/utilities/ -I$(RAVEN_DIR)/include/base/ -I$(RAVEN_DIR)/../moose/include/base/ -I$(RAVEN_DIR)/../moose/include/utils/ $(RAVEN_DIR)/python_modules/distribution1D_wrap.cxx $(RAVEN_DIR)/src/distributions/*.C $(RAVEN_DIR)/src/utilities/Interpolation_Functions.C $(RAVEN_DIR)/../moose/src/base/MooseObject.C -shared -o $(RAVEN_DIR)/python_modules/_distribution1D.so $(EXTRA_MOOSE_LIBS) $(PYTHON_LIB)


$(RAVEN_DIR)/python_modules/_raventools.so : $(RAVEN_DIR)/python_modules/raventools.i  $(RAVEN_DIR)/src/tools/batteries.C $(RAVEN_DIR)/src/tools/dieselGenerator.C $(RAVEN_DIR)/src/tools/pumpCoastdown.C $(RAVEN_DIR)/src/tools/decayHeat.C $(RAVEN_DIR)/src/tools/powerGrid.C $(RAVEN_DIR)/src/utilities/Interpolation_Functions.C
	swig -c++ -python -py3 -I$(RAVEN_DIR)/include/tools/  -I$(RAVEN_DIR)/include/utilities/ $(RAVEN_DIR)/python_modules/raventools.i
	$(libmesh_CXX) $(libmesh_CPPFLAGS) $(libmesh_CXXFLAGS) $(PYTHON_INCLUDE) -fPIC -I$(RAVEN_DIR)/include/tools/ -I$(RAVEN_DIR)/include/utilities/ $(RAVEN_DIR)/python_modules/raventools_wrap.cxx $(RAVEN_DIR)/src/tools/*.C $(RAVEN_DIR)/src/utilities/Interpolation_Functions.C -shared -o $(RAVEN_DIR)/python_modules/_raventools.so $(PYTHON_LIB)


RAVEN: $(RAVEN_APP) $(PYTHON_MODULES)

$(RAVEN_APP): $(moose_LIB) $(elk_MODULES) $(r7_LIB) $(RAVEN_LIB) $(RAVEN_app_objects)
	@echo "Linking "$@"..."
	@$(libmesh_CXX) $(libmesh_CXXFLAGS) $(RAVEN_app_objects) -o $@ $(RAVEN_LIB) $(r7_LIB) $(elk_MODULES) $(moose_LIB) $(libmesh_LIBS) $(libmesh_LDFLAGS) $(ADDITIONAL_LIBS) $(PYTHON_LIB)

-include $(RAVEN_DIR)/src/*.d
endif

delete_list := $(RAVEN_APP) $(RAVEN_LIB)

clean::
	@rm -f $(RAVEN_DIR)/python_modules/_distribution1D.so $(RAVEN_DIR)/python_modules/_raventools.so $(RAVEN_DIR)/python_modules/distribution1D_wrap.cxx $(RAVEN_DIR)/python_modules/raventools_wrap.cxx $(RAVEN_DIR)/python_modules/distribution1D.py

clobber::
	@rm -f $(RAVEN_DIR)/python_modules/_distribution1D.so $(RAVEN_DIR)/python_modules/distribution1D_wrap.cxx $(RAVEN_DIR)/python_modules/distribution1D.py

cleanall:: 
	make -C $(RAVEN_DIR) clean 
