#!/bin/bash
echo RAVEN_libs_script incoming environmental variables
env
BUILD_DIR=${BUILD_DIR:=$HOME/raven_libs/build}
INSTALL_DIR=${INSTALL_DIR:=$HOME/raven_libs/pylibs}
DOWNLOAD_DIR=${DOWNLOAD_DIR:=$BUILD_DIR/../downloads}
PYTHON_CMD=${PYTHON_CMD:=python}
export PATH="$INSTALL_DIR/bin:$PATH"
export PKG_CONFIG_PATH="$INSTALL_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"
JOBS=${JOBS:=1}
OS_NAME=`uname -sr | sed 's/\..*//'`
mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
mkdir -p $DOWNLOAD_DIR
unset OPT #OPT is sometimes defined by MOOSE, and some of the
# makefiles use it to try and determine the optimizer options.

#--insecure added to get it to work on fission computer, but should
# probably be removed at some point when running on fission is not required.
DOWNLOADER='curl -C - -L --insecure -O '

if test "$OS_NAME" = "Darwin 13"
then
    #Work around for bug in OSX.
    # The flags Apple used to compile python can't compile with Xcode 5.
    # See Xcode 5.1 release notes and
    # http://stackoverflow.com/questions/22703393
    # http://stackoverflow.com/questions/22313407
    export ARCHFLAGS=-Wno-error=unused-command-line-argument-hard-error-in-future
fi

ORIGPYTHONPATH="$PYTHONPATH"
if which shasum;
then
    SHASUM_CMD=shasum
else
    SHASUM_CMD=sha1sum
fi

update_python_path ()
{
    if ls -d $INSTALL_DIR/*/python*/site-packages/
    then
        export PYTHONPATH=`ls -d $INSTALL_DIR/*/python*/site-packages/ | tr '\n' :`:"$ORIGPYTHONPATH"
    fi
}

download_files ()
{
    ORIG_DIR=`pwd`

    SHA_SUM=$1
    URL=$2
    DL_FILENAME=`basename $URL`
    cd $DOWNLOAD_DIR
    if test -f $DL_FILENAME; then
        NEW_SHA_SUM=`$SHASUM_CMD $DL_FILENAME | cut -d " " -f 1`
    else
        NEW_SHA_SUM=no_file
    fi
    if test $SHA_SUM = $NEW_SHA_SUM; then
        echo $DL_FILENAME already downloaded
    else
        rm -f $DL_FILENAME
        $DOWNLOADER $URL
        if test -f $DL_FILENAME; then
            NEW_SHA_SUM=`$SHASUM_CMD $DL_FILENAME | cut -d " " -f 1`
        else
            NEW_SHA_SUM=no_file
        fi
        if test $SHA_SUM != $NEW_SHA_SUM; then
            echo Download of $URL failed
        else
            echo Download of $URL succeeded
        fi
    fi
    cd $ORIG_DIR
}

get_blas_and_lpack ()
{
    cd $BUILD_DIR
    download_files a643b737c30a0a5b823e11e33c9d46a605122c61 http://www.netlib.org/blas/blas.tgz
    echo Extracting blas
    tar -xzf $DOWNLOAD_DIR/blas.tgz
    export BLAS_SRC=$BUILD_DIR/BLAS
    download_files 93a6e4e6639aaf00571d53a580ddc415416e868b http://www.netlib.org/lapack/lapack-3.4.2.tgz
    echo Extracting lapack
    tar -xzf $DOWNLOAD_DIR/lapack-3.4.2.tgz
    export LAPACK_SRC=$BUILD_DIR/lapack-3.4.2
}

update_python_path

if which pcre-config;
then
    echo Perl Compatible Regular Expressions already installed
else
    cd $BUILD_DIR
    download_files a10e0040475644bfc97f7d0c0556988acfc52c6f http://downloads.sourceforge.net/project/pcre/pcre/8.35/pcre-8.35.tar.bz2
    echo Extracting pcre
    tar -xjf $DOWNLOAD_DIR/pcre-8.35.tar.bz2
    cd pcre-8.35/
    ./configure --prefix=$INSTALL_DIR
    make -j $JOBS
    make install
fi

if which swig;
then
    echo swig already installed
else
    cd $BUILD_DIR
    download_files 4203c68f79012a2951f542018ff4358d838b5035 http://downloads.sourceforge.net/project/swig/swig/swig-2.0.12/swig-2.0.12.tar.gz
    echo Extracting swig
    tar -xzf $DOWNLOAD_DIR/swig-2.0.12.tar.gz
    cd swig-2.0.12/
    ./configure --prefix=$INSTALL_DIR
    make -j $JOBS
    make install
fi

if which glibtool || libtool --version | grep 'GNU libtool';
then
    echo glibtool already installed
else
    cd $BUILD_DIR
    download_files 149e9d7a993b643d13149a94d07bbca1085e601c http://mirrors.kernel.org/gnu/libtool/libtool-2.4.tar.gz
    echo Extracting libtool
    tar -xzf $DOWNLOAD_DIR/libtool-2.4.tar.gz
    cd libtool-2.4/
    ./configure --prefix=$INSTALL_DIR --program-prefix=g
    make -j $JOBS
    make install
fi

if $PYTHON_CMD -c 'import setuptools';
then
    echo setuptools already built
else
    #setuptools
    cd $BUILD_DIR
    download_files 3e4a325d807eb0104e98985e7bd9f1ef86fc2efa https://pypi.python.org/packages/source/s/setuptools/setuptools-2.1.tar.gz
    echo Extracting setuptools
    tar -xzf $DOWNLOAD_DIR/setuptools-2.1.tar.gz
    cd setuptools-2.1
    $PYTHON_CMD setup.py install --prefix=$INSTALL_DIR
fi

if $PYTHON_CMD -c 'import numpy,sys;sys.exit(not numpy.version.version > "1.7")'
then
    echo numpy module already built
else
#numpy
#no dependencies
    get_blas_and_lpack

    download_files ba328985f20390b0f969a5be2a6e1141d5752cf9 http://downloads.sourceforge.net/project/numpy/NumPy/1.7.0/numpy-1.7.0.tar.gz
    echo Extracting numpy
    tar -xzf $DOWNLOAD_DIR/numpy-1.7.0.tar.gz
    cd numpy-1.7.0
    (unset CC CXX OPT; $PYTHON_CMD setup.py install --prefix=$INSTALL_DIR)
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
    download_files 8414ca0e6ff7d08e423955960d641ec5f309a55f http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.12/src/hdf5-1.8.12.tar.bz2
    #download_files 712955025f03db808f000d8f4976b8df0c0d37b5 http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.13.tar.bz2
    echo Extracting hdf5
    tar -xjf $DOWNLOAD_DIR/hdf5-1.8.12.tar.bz2
    cd hdf5-1.8.12
    pwd; ls -l
    (unset CC CXX FC PARALLEL OPT; ./configure --prefix=$INSTALL_DIR)
    make -j $JOBS
    make install


#cython
#no dependencies
    cd $BUILD_DIR
    download_files 03e18d5551ece9b4e3a43d4d96ad9f98c5cf6c43 http://www.cython.org/release/Cython-0.18.tar.gz
    echo Extracting Cython
    tar -xzf $DOWNLOAD_DIR/Cython-0.18.tar.gz
    cd Cython-0.18
#Python works badly with mpicc and mpicxx
    (unset CC CXX OPT; $PYTHON_CMD setup.py install --prefix=$INSTALL_DIR)

#h5py
#depends on numpy, hdf5, cython
    cd $BUILD_DIR
    download_files 4b511ed7aa28ac4c61188a121d42f17f3096c15a https://pypi.python.org/packages/source/h/h5py/h5py-2.2.1.tar.gz
    echo Extracting h5py
    tar -xzf $DOWNLOAD_DIR/h5py-2.2.1.tar.gz
    cd h5py-2.2.1
    if test "$OS_NAME" = "Darwin 13"
    then
        $PYTHON_CMD setup.py build --hdf5=$INSTALL_DIR
    else
        (unset CC CXX OPT; $PYTHON_CMD setup.py build --hdf5=$INSTALL_DIR)
    fi
    (unset CC CXX OPT; $PYTHON_CMD setup.py install --prefix=$INSTALL_DIR --hdf5=$INSTALL_DIR )
fi


if $PYTHON_CMD -c 'import scipy,sys;sys.exit(not scipy.__version__ > "0.12")'
then
    echo scipy module already built
else
#scipy
#depends on numpy
    get_blas_and_lpack
    cd $BUILD_DIR
    download_files 1ba2e2fc49ba321f62d6f78a5351336ed2509af3 http://downloads.sourceforge.net/project/scipy/scipy/0.12.0/scipy-0.12.0.tar.gz
    echo Extracting scipy
    tar -xzf $DOWNLOAD_DIR/scipy-0.12.0.tar.gz
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

if $PYTHON_CMD -c 'import sklearn,sys;sys.exit(not sklearn.__version__ > "0.14")'
then
    echo sklearn module already built
else
#scikit
#depends on numpy, scipy
    cd $BUILD_DIR
    #download_files f06a15abb107fecf7b58ef0a7057444e2d7f1369 https://pypi.python.org/packages/source/s/scikit-learn/scikit-learn-0.13.1.tar.gz
    download_files 98128859b75e3c82c995cb7524e9dbd49c1a3d9f https://pypi.python.org/packages/source/s/scikit-learn/scikit-learn-0.14.1.tar.gz
    echo Extracting scikit-learn
    tar -xzf $DOWNLOAD_DIR/scikit-learn-0.14.1.tar.gz
    cd scikit-learn-0.14.1
    if test "$OS_NAME" = "Darwin 13"
    then
        ($PYTHON_CMD setup.py install --prefix=$INSTALL_DIR)
    else
        (unset CC CXX OPT; $PYTHON_CMD setup.py install --prefix=$INSTALL_DIR)
    fi
fi

if $PYTHON_CMD -c 'import matplotlib,sys; sys.exit(not matplotlib.__version__ >= "1.4")'
then
    echo matplotlib module already built
else
#freetype
#no dependencies
    cd $BUILD_DIR
    download_files 382479336faefbc77e4b63c9ce4a96cf5d2c3585 http://downloads.sourceforge.net/project/freetype/freetype2/2.4.12/freetype-2.4.12.tar.bz2
    echo Extracting freetype
    tar -xjf $DOWNLOAD_DIR/freetype-2.4.12.tar.bz2
    cd freetype-2.4.12
    (unset CC CXX; ./configure --prefix=$INSTALL_DIR)
    make -j $JOBS
    make install

if which libpng-config
then
    echo libpng already installed
else
#libpng
#no dependencies
    cd $BUILD_DIR
    download_files 6bcd6efa7f20ccee51e70453426d7f4aea7cf4bb http://download.sourceforge.net/libpng/libpng-1.6.12.tar.gz
    echo Extracting libpng
    tar -xzf $DOWNLOAD_DIR/libpng-1.6.12.tar.gz
    cd libpng-1.6.12
    (unset CC CXX; ./configure --prefix=$INSTALL_DIR)
    make -j $JOBS
    make install
fi

#git matplotlib
#depends on numpy, freetype, a patch file, png
   cd $BUILD_DIR
#   git clone https://github.com/matplotlib/matplotlib.git
#   cd matplotlib; git checkout v1.4.x
   download_files bdd84b713290207b108343c8af37ea25c8e2aadb https://downloads.sourceforge.net/project/matplotlib/matplotlib/matplotlib-1.4.0/matplotlib-1.4.0.tar.gz
   tar -xzf $DOWNLOAD_DIR/matplotlib-1.4.0.tar.gz
   cd matplotlib-1.4.0
   #download_files befdcf1229163277439dccc00bd5be04685229e4 https://github.com/matplotlib/matplotlib/archive/v1.4.0rc1.tar.gz
   #tar -xzf $DOWNLOAD_DIR/v1.4.0rc1.tar.gz
   #cd matplotlib-1.4.0rc1
   #sed -i -e "s/default_libraries=\['png', 'z'\])/default_libraries=\['png', 'z'\], alt_exec='libpng-config --ldflags')/g" setupext.py
   (unset CC CXX; $PYTHON_CMD setup.py install --prefix=$INSTALL_DIR)

#matplotlib
#depends on numpy, freetype
#    cd $BUILD_DIR
#    download_files 8578afc86424392591c0ee03f7613ffa9b6f68ee http://downloads.sourceforge.net/project/matplotlib/matplotlib/matplotlib-1.3.1/matplotlib-1.3.1.tar.gz
#    echo Extracting matplotlib
#    tar -xzf $DOWNLOAD_DIR/matplotlib-1.3.1.tar.gz
#    cd matplotlib-1.3.1
#    (unset CC CXX; $PYTHON_CMD setup.py install --prefix=$INSTALL_DIR)

fi

update_python_path

#boost

# if test ! -e $INSTALL_DIR/include/boost/random/mersenne_twister.hpp
# then
#     cd $BUILD_DIR
#     download_files 61ed0e57d3c7c8985805bb0682de3f4c65f4b6e5  http://downloads.sourceforge.net/project/boost/boost/1.55.0/boost_1_55_0.tar.gz
#     tar -xzf $DOWNLOAD_DIR/boost_1_55_0.tar.gz
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
