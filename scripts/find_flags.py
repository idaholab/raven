#!/usr/bin/env python3
import distutils.sysconfig
import sys
if len(sys.argv) < 2:
    print(sys.argv[0],"[library] [include]")
    sys.exit()

if "include" in sys.argv:
    include_dir = distutils.sysconfig.get_python_inc()
    print("-I",include_dir,sep='',end=' ')

if "library" in sys.argv:
    lib_dir = distutils.sysconfig.get_config_var('LIBDIR')
    library_name = distutils.sysconfig.get_config_var('LDLIBRARY')
    static = False
    if library_name.startswith("lib"):
        library_name = library_name[3:]
    if library_name.endswith(".so"):
        library_name = library_name[:-3]
    if library_name.endswith(".dylib"):
        library_name = library_name[:-6]
    if library_name.endswith(".a"):
        library_name = library_name[:-2]
        static = True
    if library_name.startswith("Python.framework"):
        #This must be a mac
        version = library_name.split("/")[2]
        library_name = "python"+version
    extra = ""
    if static:
        extra = " "+distutils.sysconfig.get_config_var('LIBS')+\
            " "+distutils.sysconfig.get_config_var('SYSLIBS')+\
            " -Xlinker -export-dynamic"
    print("-L",lib_dir," -l",library_name,extra,sep='',end=' ')


#Note how to do this was basically found by looking at the code that was 
#executed when the following was run:
#from distutils.core import setup, Extension
#setup(name="example",
#      version="1.0",
#      ext_modules=[Extension('example',["example.cxx","example_wrap.cxx"])]
#)
