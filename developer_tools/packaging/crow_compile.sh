#!/bin/bash
INSTALL_DIR=${INSTALL_DIR:=$HOME/raven_libs/pylibs}
BUILD_DIR=${BUILD_DIR:=$HOME/raven_libs/build}
PYTHON_CMD=${PYTHON_CMD:=python}
SCRIPT_DIR=`dirname $0`
RAVEN_DIR=`cd $SCRIPT_DIR/../../; pwd`
CROW_OVER_DIR=$RAVEN_DIR/../crow
CROW_UP_DIR=$RAVEN_DIR/crow
if test -e $CROW_OVER_DIR; then
    CROW_DIR=$CROW_OVER_DIR
else
    if test -e $CROW_UP_DIR; then
        CROW_DIR=$CROW_UP_DIR
    else
        echo crow directory not found at $CROW_OVER_DIR or $CROW_UP_DIR
    fi
fi
#echo $RAVEN_DIR $CROW_DIR $INSTALL_DIR
PATH="$PATH:$INSTALL_DIR/bin"
cd $CROW_DIR
$PYTHON_CMD setup.py build_ext build --build-base=$BUILD_DIR install --prefix=$INSTALL_DIR
cd $RAVEN_DIR
$PYTHON_CMD setup.py build_ext build --build-base=$BUILD_DIR install --prefix=$INSTALL_DIR
