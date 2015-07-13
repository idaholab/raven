#!/bin/bash
BUILD_DIR=${BUILD_DIR:=$HOME/raven_libs/build}
INSTALL_DIR=${INSTALL_DIR:=$HOME/raven_libs/pylibs}
DOWNLOAD_DIR=${DOWNLOAD_DIR:=$BUILD_DIR/../downloads}
PYTHON_CMD=${PYTHON_CMD:=python}
JOBS=${JOBS:=1}
OS_NAME=`uname -sr | sed 's/\..*//'`
mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
mkdir -p $DOWNLOAD_DIR
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
        if test $SHA_SUM != `$SHASUM_CMD $DL_FILENAME | cut -d " " -f 1`; then 
            echo Download of $URL failed
        else
            echo Download of $URL succeeded
        fi
    fi
    cd $ORIG_DIR
}

PATH="$PATH:$INSTALL_DIR/bin"

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


if which glibtool;
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
