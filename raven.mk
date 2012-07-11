RAVEN_SRC_DIRS := $(RAVEN_DIR)/src/*/*

PYTHON3_HELLO = $(shell python3 -c "print('HELLO')")

SWIG_VERSION = $(shell swig -version)


ifeq ($(PYTHON3_HELLO),HELLO)
	PYTHON_INCLUDE = $(shell $(RAVEN_DIR)/scripts/find_flags.py include) #-DPy_LIMITED_API
	PYTHON_LIB = $(shell $(RAVEN_DIR)/scripts/find_flags.py library) #-DPy_LIMITED_API
ifeq ($(findstring SWIG Version 2,$(SWIG_VERSION)),)
	PYTHON_MODULES = 
else
	PYTHON_MODULES = $(RAVEN_DIR)/python_modules/_distribution1D.so
endif

else
#Python3 not found.
	PYTHON_INCLUDE = -DNO_PYTHON3_FOR_YOU
	PYTHON_LIB = -DNO_PYTHON3_FOR_YOU
	PYTHON_MODULES = 
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

all:: $(RAVEN_LIB)

# build rule for lib RAVEN
ifeq ($(enable-shared),yes)
# Build dynamic library
$(RAVEN_LIB): $(RAVEN_objects)
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

$(RAVEN_DIR)/python_modules/_distribution1D.so : $(RAVEN_DIR)/python_modules/distribution1D.i  $(RAVEN_DIR)/src/distributions/distribution_1D.C $(RAVEN_DIR)/src/distributions/DistributionContainer.C
	swig -c++ -python -py3 -I$(RAVEN_DIR)/include/distributions/ $(RAVEN_DIR)/python_modules/distribution1D.i
	$(libmesh_CXX) $(libmesh_CPPFLAGS) $(libmesh_CXXFLAGS) $(PYTHON_INCLUDE) -fPIC -I$(RAVEN_DIR)/include/distributions/  $(RAVEN_DIR)/python_modules/distribution1D_wrap.cxx $(RAVEN_DIR)/src/distributions/*.C -shared -o $(RAVEN_DIR)/python_modules/_distribution1D.so $(PYTHON_LIB)

RAVEN: $(RAVEN_APP) $(PYTHON_MODULES)

$(RAVEN_APP): $(moose_LIB) $(elk_MODULES) $(r7_LIB) $(RAVEN_LIB) $(RAVEN_app_objects)
	@echo "Linking "$@"..."
	@$(libmesh_CXX) $(libmesh_CXXFLAGS) $(RAVEN_app_objects) -o $@ $(RAVEN_LIB) $(r7_LIB) $(elk_MODULES) $(moose_LIB) $(libmesh_LIBS) $(libmesh_LDFLAGS) $(ADDITIONAL_LIBS) $(PYTHON_LIB)

-include $(RAVEN_DIR)/src/*.d
endif


clean::
	@rm -fr $(RAVEN_APP)
	@rm -fr $(RAVEN_LIB)
	@find . \( -name "*~" -or -name "*.o" -or -name "*.d" -or -name "*.pyc" \) -exec rm '{}' \;
	@rm -fr *.mod
	@rm -f $(RAVEN_DIR)/python_modules/_distribution1D.so $(RAVEN_DIR)/python_modules/distribution1D_wrap.cxx $(RAVEN_DIR)/python_modules/distribution1D.py

clobber::
	@rm -fr $(RAVEN_APP)
	@rm -fr $(RAVEN_LIB)
	@find . \( -name "*~" -or -name "*.o" -or -name "*.d" -or -name "*.pyc" \
                -or -name "*.gcda" -or -name "*.gcno" -or -name "*.gcov" \) -exec rm '{}' \;
	@rm -fr *.mod

cleanall::
	make -C $(RAVEN_DIR) clean
