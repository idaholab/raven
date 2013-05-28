BUILD_DIR=${BUILD_DIR:=$HOME/raven_libs/build}
INSTALL_DIR=${INSTALL_DIR:=$HOME/raven_libs/pylibs}
mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
export PYTHONPATH=$INSTALL_DIR/lib/python2.7/site-packages/
cd $BUILD_DIR

#hdf5
curl -v -O http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.11.tar.bz2
tar -xvjf hdf5-1.8.11.tar.bz2
cd hdf5-1.8.11
(unset CC CXX F90 F77 FC; ./configure --prefix=$INSTALL_DIR)
make
make install

#cython
cd $BUILD_DIR
curl -O http://www.cython.org/release/Cython-0.18.tar.gz
tar -xvzf Cython-0.18.tar.gz
cd Cython-0.18
#Python works badly with mpicc and mpicxx
(unset CC CXX; python setup.py install --prefix=$INSTALL_DIR)

#numpy
cd $BUILD_DIR
curl -O http://iweb.dl.sourceforge.net/project/numpy/NumPy/1.7.0/numpy-1.7.0.tar.gz
tar -xvzf numpy-1.7.0.tar.gz
cd numpy-1.7.0
(unset CC CXX; python setup.py install --prefix=$INSTALL_DIR)

#h5py
cd $BUILD_DIR
curl -O http://h5py.googlecode.com/files/h5py-2.1.2.tar.gz
tar -xvzf h5py-2.1.2.tar.gz
cd h5py-2.1.2
(unset CC CXX; python setup.py build --hdf5=$INSTALL_DIR)
(unset CC CXX; python setup.py install --prefix=$INSTALL_DIR)

#scipy
cd $BUILD_DIR
curl -O http://iweb.dl.sourceforge.net/project/scipy/scipy/0.12.0/scipy-0.12.0.tar.gz
tar -xvzf scipy-0.12.0.tar.gz
cd scipy-0.12.0
(unset CC CXX F90 F77 FC; python setup.py install --prefix=$INSTALL_DIR)

#scikit
cd $BUILD_DIR
curl -O https://pypi.python.org/packages/source/s/scikit-learn/scikit-learn-0.13.1.tar.gz
tar -xvzf scikit-learn-0.13.1.tar.gz
cd scikit-learn-0.13.1
(unset CC CXX; python setup.py install --prefix=$INSTALL_DIR)