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

#Update path
PATH="$INSTALL_DIR/bin:$PATH"

#PyYAML
if $PYTHON_CMD -c 'import yaml'
then
    echo yaml module already built
else
    cd $BUILD_DIR
    download_files PyYAML-3.10.tar.gz 476dcfbcc6f4ebf3c06186229e8e2bd7d7b20e73 http://pyyaml.org/download/pyyaml/PyYAML-3.10.tar.gz
    tar -xvzf PyYAML-3.10.tar.gz
    cd PyYAML-3.10
    (unset CC CXX; $PYTHON_CMD setup.py install --prefix=$INSTALL_DIR)
fi

#Qt
if which qmake
then
    echo Qt already installed
else
    cd $BUILD_DIR
    download_files qt-everywhere-opensource-src-4.8.5.tar.gz 745f9ebf091696c0d5403ce691dc28c039d77b9e http://download.qt-project.org/official_releases/qt/4.8/4.8.5/qt-everywhere-opensource-src-4.8.5.tar.gz
    tar -xvzf qt-everywhere-opensource-src-4.8.5.tar.gz
    cd qt-everywhere-opensource-src-4.8.5
    ./configure --prefix=$INSTALL_DIR -opensource -confirm-license
    make -j $JOBS
    make install 
fi

#try and figure out python directory for sip and PyQt4

if ls -d $INSTALL_DIR/*/python*/site-packages/
then
    PYTHON_SITE_PACKAGES=`ls -d $INSTALL_DIR/*/python*/site-packages/`
else
    TAIL=`$PYTHON_CMD -c 'import sys,distutils.sysconfig; prefix = distutils.sysconfig.get_config_var("prefix"); libdest = distutils.sysconfig.get_config_var("LIBDEST"); sys.stdout.write(libdest[len(prefix):])'`
    PYTHON_SITE_PACKAGES="$INSTALL_DIR/$TAIL/site-packages"
fi

#sip

if $PYTHON_CMD -c 'import sipconfig'
then
    echo sip built already
else
    cd $BUILD_DIR
    download_files sip-4.14.7.tar.gz ee048f6db7257d1eae2d9d2e407c1657c8888023 http://sourceforge.net/projects/pyqt/files/sip/sip-4.14.7/sip-4.14.7.tar.gz
    tar -xvzf sip-4.14.7.tar.gz
    cd sip-4.14.7
    $PYTHON_CMD configure.py -b $INSTALL_DIR/bin -d $PYTHON_SITE_PACKAGES  -v $INSTALL_DIR/share/sip -e $INSTALL_DIR/include
    make -j $JOBS
    make install
fi

update_python_path

#PyQt4


if $PYTHON_CMD -c 'import PyQt4.QtCore'
then
    echo PyQt4 build already
else
    cd $BUILD_DIR
    download_files PyQt-mac-gpl-4.10.2.tar.gz 40362e6b9f476683e4e35b83369e30a8dfff99ad http://sourceforge.net/projects/pyqt/files/PyQt4/PyQt-4.10.2/PyQt-mac-gpl-4.10.2.tar.gz
    tar -xvzf PyQt-mac-gpl-4.10.2.tar.gz
    cd PyQt-mac-gpl-4.10.2
    $PYTHON_CMD configure.py -b $INSTALL_DIR/bin -d $PYTHON_SITE_PACKAGES -p $INSTALL_DIR/plugins -v $INSTALL_DIR/share/sip --confirm-license
    make -j $JOBS
    make install
fi

update_python_path

#cmake

if which cmake
then
    echo cmake already built
else
    cd $BUILD_DIR
    download_files cmake-2.8.10.2.tar.gz 2d868ccc3f9f2aa7c2844bd0a4609d5313edaaec http://www.cmake.org/files/v2.8/cmake-2.8.10.2.tar.gz
    tar -xvzf cmake-2.8.10.2.tar.gz
    cd cmake-2.8.10.2
    ./configure --prefix=$INSTALL_DIR
    make -j $JOBS
    make install
fi 

#VTK
if $PYTHON_CMD -c 'import vtk'
then
    echo VTK build already
else
    cd $BUILD_DIR
    #VTK's download is messed up
    #download_files vtk-5.10.1.tar.gz  264b0052e65bd6571a84727113508789 http://www.vtk.org/files/release/5.10/vtk-5.10.1.tar.gz
    download_files vtk-5.10.1.zip ffc01be273e00194d1f0051837a5eadc52bb0d20 http://www.vtk.org/files/release/5.10/vtk-5.10.1.zip
    rm -Rvf VTK5.10.1
    unzip vtk-5.10.1.zip
    mkdir VTK-build
    cd VTK-build
    cmake ../VTK5.10.1 -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        -DBUILD_SHARED_LIBS=ON \
        -DVTK_WRAP_PYTHON=ON \
        -DVTK_WRAP_PYTHON_SIP=ON \
        -DVTK_USE_QT=ON \
        -DVTK_USE_QVTK_QTOPENGL=ON \
        -DSIP_INCLUDE_DIR=$INSTALL_DIR/include \
        -DSIP_PYQT_DIR=$INSTALL_DIR/share/sip \
        -DVTK_LEGACY_REMOVE=ON \
        -DVTK_USE_TK=OFF
    make -j $JOBS
    make install
    cd $BUILD_DIR
    rm -Rvf VTK-build
fi

update_python_path
