#!/bin/bash
BUILD_DIR=${BUILD_DIR:=$HOME/raven_libs/build}
INSTALL_DIR=${INSTALL_DIR:=$HOME/raven_libs/pylibs}
PYTHON_CMD=${PYTHON_CMD:=python}
JOBS=${JOBS:=1}
mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
DOWNLOADER='curl -C - -L -O '

ORIGPYTHONPATH="$PYTHONPATH"

update_python_path ()
{
    if ls -d $INSTALL_DIR/*/python*/site-packages/
    then
        export PYTHONPATH=`ls -d $INSTALL_DIR/*/python*/site-packages/ | tr '\n' :`:"$ORIGPYTHONPATH"
    fi
}

download_files ()
{
    DL_FILENAME=$1 
    SHA_SUM=$2
    URL=$3
    if test -f $DL_FILENAME; then
        NEW_SHA_SUM=`shasum $DL_FILENAME | cut -d " " -f 1`
    else
        NEW_SHA_SUM=no_file
    fi
    if test $SHA_SUM = $NEW_SHA_SUM; then 
        echo $DL_FILENAME already downloaded
    else
        rm -f $DL_FILENAME
        $DOWNLOADER $URL
        if test $SHA_SUM != `shasum $DL_FILENAME | cut -d " " -f 1`; then 
            echo Download of $URL failed
        else
            echo Download of $URL succeeded 
        fi
    fi
}

if curl http://www.energy.gov > /dev/null
then
    echo Successfully got data from the internet
else
    echo Could not connect to internet
    echo Enter Proxy Username
    read username
    echo Enter Proxy Password
    read password
    export http_proxy="http://${username}:${password}@134.20.11.87:8080"
    export https_proxy=$http_proxy
    if curl http://www.energy.gov > /dev/null
    then
        echo Successfully got data from the internet
    else
        echo Proxy setting did not help
    fi

fi

update_python_path

if $PYTHON_CMD -c 'import numpy,sys;sys.exit(not numpy.version.version > "1.7")'
then 
    echo numpy module already built
else
#numpy
#no dependencies
    cd $BUILD_DIR
    download_files blas.tgz a643b737c30a0a5b823e11e33c9d46a605122c61 http://www.netlib.org/blas/blas.tgz
    tar -xvzf blas.tgz
    export BLAS_SRC=$BUILD_DIR/BLAS
    download_files lapack-3.4.2.tgz 93a6e4e6639aaf00571d53a580ddc415416e868b http://www.netlib.org/lapack/lapack-3.4.2.tgz
    tar -xvzf lapack-3.4.2.tgz
    export LAPACK_SRC=$BUILD_DIR/lapack-3.4.2
    download_files numpy-1.7.0.tar.gz ba328985f20390b0f969a5be2a6e1141d5752cf9 http://downloads.sourceforge.net/project/numpy/NumPy/1.7.0/numpy-1.7.0.tar.gz
    tar -xvzf numpy-1.7.0.tar.gz
    cd numpy-1.7.0
    (unset CC CXX; $PYTHON_CMD setup.py install --prefix=$INSTALL_DIR)
fi

update_python_path
#export PYTHONPATH=`ls -d $INSTALL_DIR/*/python*/site-packages/`:"$ORIGPYTHONPATH"

if $PYTHON_CMD -c 'import h5py'
then
    echo h5py module already built
else
#hdf5
#no dependencies
    cd $BUILD_DIR
    rm -Rvf hdf5-1.8.12
    download_files hdf5-1.8.12.tar.bz2 8414ca0e6ff7d08e423955960d641ec5f309a55f http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.12.tar.bz2
    tar -xvjf hdf5-1.8.12.tar.bz2
    cd hdf5-1.8.12
    pwd; ls -l
    (unset CC CXX FC PARALLEL; ./configure --prefix=$INSTALL_DIR)
    make -j $JOBS
    make install


#cython
#no dependencies
    cd $BUILD_DIR
    download_files Cython-0.18.tar.gz 03e18d5551ece9b4e3a43d4d96ad9f98c5cf6c43 http://www.cython.org/release/Cython-0.18.tar.gz
    tar -xvzf Cython-0.18.tar.gz
    cd Cython-0.18
#Python works badly with mpicc and mpicxx
    (unset CC CXX; $PYTHON_CMD setup.py install --prefix=$INSTALL_DIR)

#h5py
#depends on numpy, hdf5, cython
    cd $BUILD_DIR
    download_files h5py-2.2.0.tar.gz 65e5d6cc83d9c1cb562265a77a46def22e9e6593 http://h5py.googlecode.com/files/h5py-2.2.0.tar.gz
    tar -xvzf h5py-2.2.0.tar.gz
    cd h5py-2.2.0
    (unset CC CXX; $PYTHON_CMD setup.py build --hdf5=$INSTALL_DIR)
    (unset CC CXX; $PYTHON_CMD setup.py install --prefix=$INSTALL_DIR)  
fi


if $PYTHON_CMD -c 'import scipy'
then
    echo scipy module already built
else
#scipy
#depends on numpy
    cd $BUILD_DIR
    download_files scipy-0.12.0.tar.gz 1ba2e2fc49ba321f62d6f78a5351336ed2509af3 http://downloads.sourceforge.net/project/scipy/scipy/0.12.0/scipy-0.12.0.tar.gz
    tar -xvzf scipy-0.12.0.tar.gz
    cd scipy-0.12.0
    patch -p1 << PATCH_SCIPY
--- scipy-0.12.0/scipy/_build_utils/_fortran.py	2013-04-06 10:10:34.000000000 -0600
+++ scipy-0.12.0_mod/scipy/_build_utils/_fortran.py	2013-07-31 13:51:13.965027409 -0600
@@ -16,8 +16,11 @@
 
     libraries = info.get('libraries', '')
     for library in libraries:
-        if r_mkl.search(library):
-            return True
+        try:
+            if r_mkl.search(library):
+                return True
+        except:
+            pass
 
     return False
 
PATCH_SCIPY
    (unset CC CXX F90 F77 FC; $PYTHON_CMD setup.py install --prefix=$INSTALL_DIR)
fi

update_python_path

if $PYTHON_CMD -c 'import sklearn'
then
    echo sklearn module already built
else
#scikit
#depends on numpy, scipy
    cd $BUILD_DIR
    #download_files scikit-learn-0.13.1.tar.gz f06a15abb107fecf7b58ef0a7057444e2d7f1369 https://pypi.python.org/packages/source/s/scikit-learn/scikit-learn-0.13.1.tar.gz
    download_files scikit-learn-0.14.1.tar.gz 98128859b75e3c82c995cb7524e9dbd49c1a3d9f https://pypi.python.org/packages/source/s/scikit-learn/scikit-learn-0.14.1.tar.gz 
    tar -xvzf scikit-learn-0.14.1.tar.gz
    cd scikit-learn-0.14.1
    if test "`uname -sr | sed 's/\..*//'`" = "Darwin 13"
    then
	($PYTHON_CMD setup.py install --prefix=$INSTALL_DIR)
    else
	(unset CC CXX; $PYTHON_CMD setup.py install --prefix=$INSTALL_DIR)
    fi
fi

if $PYTHON_CMD -c 'import matplotlib,sys; sys.exit(not matplotlib.__version__ > "1.3")'
then 
    echo matplotlib module already built
else
#freetype
#no dependencies
    cd $BUILD_DIR
    download_files freetype-2.4.12.tar.bz2 382479336faefbc77e4b63c9ce4a96cf5d2c3585 http://downloads.sourceforge.net/project/freetype/freetype2/2.4.12/freetype-2.4.12.tar.bz2
    tar -xvjf freetype-2.4.12.tar.bz2
    cd freetype-2.4.12
    (unset CC CXX; ./configure --prefix=$INSTALL_DIR)
    make -j $JOBS
    make install

#matplotlib
#depends on numpy, freetype
    cd $BUILD_DIR
    download_files matplotlib-1.3.1.tar.gz 8578afc86424392591c0ee03f7613ffa9b6f68ee http://downloads.sourceforge.net/project/matplotlib/matplotlib/matplotlib-1.3.1/matplotlib-1.3.1.tar.gz
    tar -xvzf matplotlib-1.3.1.tar.gz
    cd matplotlib-1.3.1
    (unset CC CXX; $PYTHON_CMD setup.py install --prefix=$INSTALL_DIR)
    
fi

update_python_path

#boost

# if test ! -e $INSTALL_DIR/include/boost/random/mersenne_twister.hpp
# then
#     cd $BUILD_DIR
#     download_files boost_1_55_0.tar.gz 61ed0e57d3c7c8985805bb0682de3f4c65f4b6e5  http://downloads.sourceforge.net/project/boost/boost/1.55.0/boost_1_55_0.tar.gz
#     tar -xvzf boost_1_55_0.tar.gz
#     mkdir -p $INSTALL_DIR/include
#     cp -Rp boost_1_55_0/boost $INSTALL_DIR/include
# else
#     echo boost already installed
# fi


$PYTHON_CMD <<PYTHON_SCRIPT
from __future__ import print_function
l = ["numpy","h5py","scipy","sklearn","matplotlib"]
found = []
notfound = []
for i in l:
  try:
    print(__import__(i))
    found.append(i)
  except:
    notfound.append(i)

print("Found Modules: ",found)
print("Not Found Modules: ",notfound)
PYTHON_SCRIPT

echo $PYTHON_CMD PYTHONPATH=$PYTHONPATH
