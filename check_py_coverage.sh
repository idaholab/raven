#!/bin/bash
BUILD_DIR=${BUILD_DIR:=$HOME/raven_libs/build}
INSTALL_DIR=${INSTALL_DIR:=$HOME/raven_libs/pylibs}
PYTHON_CMD=${PYTHON_CMD:=python}
JOBS=${JOBS:=1}
mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
DOWNLOADER='curl -C - -L -O '
SCRIPT_DIRNAME=`dirname $0`
SCRIPT_DIR=`(cd $SCRIPT_DIRNAME; pwd)`

ORIGPYTHONPATH="$PYTHONPATH"

update_python_path ()
{
    if ls -d $INSTALL_DIR/*/python*/site-packages/
    then
        export PYTHONPATH=`ls -d $INSTALL_DIR/*/python*/site-packages/`:"$ORIGPYTHONPATH"
    fi
}

if curl http://www.energy.gov > /dev/null
then
    echo Successfully got data from the internet
else
    echo Could not connect to internet
fi

update_python_path
PATH="$INSTALL_DIR/bin:$PATH"

if which coverage
then
    echo coverage already available, skipping building it.
else
    cd $BUILD_DIR
    $DOWNLOADER https://pypi.python.org/packages/source/c/coverage/coverage-3.6.tar.gz #md5=67d4e393f4c6a5ffc18605409d2aa1ac
    tar -xvzf coverage-3.6.tar.gz
    cd coverage-3.6
    (unset CC CXX; $PYTHON_CMD setup.py install --prefix=$INSTALL_DIR)
fi

update_python_path

cd $SCRIPT_DIR

EXTRA='--source=../../framework -a'
cd tests/framework
coverage erase
for I in test_simple.xml test_branch.xml test_mpi.xml test_output.xml test_sbo_heavy_det.xml test_push_into_hdf5.xml test_rom_trainer.xml test_Grid_Sampler.xml test_LHS_Sampler.xml test_Lorentz.xml test_bison_mc_simple.xml
do
    coverage run $EXTRA ../../framework/Driver.py  $I
done
coverage run $EXTRA ../../framework/TestDistributions.py
coverage html

