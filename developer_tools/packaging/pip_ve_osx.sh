#!/bin/bash

SCRIPT_NAME=`readlink $0`
if test -x "$SCRIPT_NAME";
then
    SCRIPT_DIRNAME=`dirname $SCRIPT_NAME`
else
    SCRIPT_DIRNAME=`dirname $0`
fi
SCRIPT_DIR=`(cd $SCRIPT_DIRNAME; pwd)`

MACOSX_DEPLOYMENT_TARGET=10.9;
export MACOSX_DEPLOYMENT_TARGET;

BUILD_DIR="$HOME/build_raven_libs"
INSTALL_DIR=${INSTALL_DIR:=$HOME/raven_libs}
STAGE_DIR="$BUILD_DIR/staging"
DOWNLOAD_DIR=${DOWNLOAD_DIR:=$HOME/Downloads/raven_downloads}
mkdir -p $BUILD_DIR
mkdir -p $DOWNLOAD_DIR

JOBS=${JOBS:=1}
DOWNLOADER='curl -C - -L -O '

if which shasum;
then
    SHASUM_CMD=shasum
else
    SHASUM_CMD=sha1sum
fi

download_file ()
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



cd $BUILD_DIR
download_file dce2b862a30099ee48c19a7c34e2d7c2eeff5670 https://www.python.org/ftp/python/2.7.13/Python-2.7.13.tgz
tar -xzf $DOWNLOAD_DIR/Python-2.7.13.tgz
cd Python-2.7.13/
./configure --prefix="$STAGE_DIR"
make -j${JOBS}
make install

OLD_PATH="$PATH"
PATH="$STAGE_DIR/bin:$PATH"
which python

python -m ensurepip
cd $BUILD_DIR
pip install virtualenv
#virtualenv $INSTALL_DIR
#source $INSTALL_DIR/bin/activate

virtualenv --always-copy $INSTALL_DIR

PATH="$OLD_PATH"

source $INSTALL_DIR/bin/activate
#Call library_handler to return the pip install command with the qa'd versions
`python $SCRIPT_DIR/library_handler.py pip --action install`
#pip install numpy==1.11.0 h5py==2.6.0 scipy==0.17.1 scikit-learn==0.17.1 matplotlib==1.5.1


cd $BUILD_DIR
download_file a10e0040475644bfc97f7d0c0556988acfc52c6f http://downloads.sourceforge.net/project/pcre/pcre/8.35/pcre-8.35.tar.bz2
tar -xjf $DOWNLOAD_DIR/pcre-8.35.tar.bz2
cd pcre-8.35/
./configure --prefix=$INSTALL_DIR
make -j${JOBS}
make install


cd $BUILD_DIR
download_file 5cc1af41d041e4cc609580b99bb3dcf720effa25 https://downloads.sourceforge.net/project/swig/swig/swig-3.0.12/swig-3.0.12.tar.gz
tar -xzf $DOWNLOAD_DIR/swig-3.0.12.tar.gz
cd swig-3.0.12/
./configure --prefix="$INSTALL_DIR"
make -j${JOBS}
make install


#Defining MPLBACKEND, Otherwise matplotlib complains that python needs to be
# built as a framework
#However if I build python with:
#./configure --enable-framework=$HOME/python_framework --prefix=$HOME/python_unix
#I instead get: "Could not find platform independent libraries"
#TODO this should also be removed when deactivate is called.
echo 'export MPLBACKEND="TkAgg"' >> $INSTALL_DIR/bin/activate
