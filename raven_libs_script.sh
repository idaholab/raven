#!/bin/bash
BUILD_DIR=${BUILD_DIR:=$HOME/raven_libs/build}
INSTALL_DIR=${INSTALL_DIR:=$HOME/raven_libs/pylibs}
mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR

ORIGPYTHONPATH="$PYTHONPATH"

update_python_path ()
{
    if ls -d $INSTALL_DIR/*/python*/site-packages/
    then
        export PYTHONPATH=`ls -d $INSTALL_DIR/*/python*/site-packages/`:"$ORIGPYTHONPATH"
    fi
}

if python -c 'import numpy'
then 
    echo numpy module already built
else
#numpy
#no dependencies
    cd $BUILD_DIR
    curl -O http://iweb.dl.sourceforge.net/project/numpy/NumPy/1.7.0/numpy-1.7.0.tar.gz
    tar -xvzf numpy-1.7.0.tar.gz
    cd numpy-1.7.0
    (unset CC CXX; python setup.py install --prefix=$INSTALL_DIR)
fi

update_python_path
#export PYTHONPATH=`ls -d $INSTALL_DIR/*/python*/site-packages/`:"$ORIGPYTHONPATH"

if python -c 'import h5py'
then
    echo h5py module already built
else
#hdf5
#no dependencies
    cd $BUILD_DIR
    curl -v -O http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.11.tar.bz2
    tar -xvjf hdf5-1.8.11.tar.bz2
    cd hdf5-1.8.11
    (unset CC CXX F90 F77 FC; ./configure --prefix=$INSTALL_DIR)
    make
    make install

#cython
#no dependencies
    cd $BUILD_DIR
    curl -O http://www.cython.org/release/Cython-0.18.tar.gz
    tar -xvzf Cython-0.18.tar.gz
    cd Cython-0.18
#Python works badly with mpicc and mpicxx
    (unset CC CXX; python setup.py install --prefix=$INSTALL_DIR)

#h5py
#depends on numpy, hdf5, cython
    cd $BUILD_DIR
    curl -O http://h5py.googlecode.com/files/h5py-2.1.2.tar.gz
    tar -xvzf h5py-2.1.2.tar.gz
    cd h5py-2.1.2
    (unset CC CXX; python setup.py build --hdf5=$INSTALL_DIR)
    (unset CC CXX; python setup.py install --prefix=$INSTALL_DIR)  
fi


if python -c 'import scipy'
then
    echo scipy module already built
else
#scipy
#depends on numpy
    cd $BUILD_DIR
    curl -O http://iweb.dl.sourceforge.net/project/scipy/scipy/0.12.0/scipy-0.12.0.tar.gz
    tar -xvzf scipy-0.12.0.tar.gz
    cd scipy-0.12.0
    (unset CC CXX F90 F77 FC; python setup.py install --prefix=$INSTALL_DIR)
fi

update_python_path

if python -c 'import sklearn'
then
    echo sklearn module already built
else
#scikit
#depends on numpy, scipy
    cd $BUILD_DIR
    curl -O https://pypi.python.org/packages/source/s/scikit-learn/scikit-learn-0.13.1.tar.gz
    tar -xvzf scikit-learn-0.13.1.tar.gz
    cd scikit-learn-0.13.1
    (unset CC CXX; python setup.py install --prefix=$INSTALL_DIR)
fi

if python -c 'import matplotlib'
then 
    echo matplotlib module already built
else
#freetype
#no dependencies
    cd $BUILD_DIR
    curl -L -O http://downloads.sourceforge.net/project/freetype/freetype2/2.4.12/freetype-2.4.12.tar.bz2
    tar -xvjf freetype-2.4.12.tar.bz2
    cd freetype-2.4.12
    (unset CC CXX; ./configure --prefix=$INSTALL_DIR)
    make
    make install

#matplotlib
#depends on numpy, freetype
    cd $BUILD_DIR
    curl -L -O https://downloads.sourceforge.net/project/matplotlib/matplotlib/matplotlib-1.2.1/matplotlib-1.2.1.tar.gz
    tar -xvzf matplotlib-1.2.1.tar.gz
    cd matplotlib-1.2.1
    (unset CC CXX; python setup.py install --prefix=$INSTALL_DIR)
    
fi

update_python_path


echo PYTHONPATH=$PYTHONPATH
