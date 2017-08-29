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

update_python_path
PATH=$INSTALL_DIR/bin:$PATH

if which coverage
then
    echo coverage already available, skipping building it.
else
    if curl http://www.energy.gov > /dev/null
    then
       echo Successfully got data from the internet
    else
       echo Could not connect to internet
    fi

    cd $BUILD_DIR
    $DOWNLOADER https://pypi.python.org/packages/source/c/coverage/coverage-3.7.1.tar.gz #md5=67d4e393f4c6a5ffc18605409d2aa1ac
    tar -xvzf coverage-3.7.1.tar.gz
    cd coverage-3.7.1
    (unset CC CXX; $PYTHON_CMD setup.py install --prefix=$INSTALL_DIR)
fi

update_python_path

cd $SCRIPT_DIR

cd tests/framework
#coverage help run
FRAMEWORK_DIR=`(cd ../../framework && pwd)`

source $SCRIPT_DIR/scripts/setup_raven_libs

EXTRA="--rcfile=$FRAMEWORK_DIR/../tests/framework/.coveragerc --source=$FRAMEWORK_DIR -a --omit=$FRAMEWORK_DIR/contrib/*"
export COVERAGE_FILE=`pwd`/.coverage

coverage erase
#skip test_rom_trainer.xml
DRIVER=$FRAMEWORK_DIR/Driver.py
for I in $(python ${SCRIPT_DIR}/developer_tools/get_coverage_tests.py)
do
    #DIR=`dirname $I`
    #BASE=`basename $I`
    #echo Running $DIR $BASE
    #(cd $DIR && coverage run $EXTRA $DRIVER $BASE)
    echo coverage run $EXTRA $DRIVER $I
    coverage run $EXTRA $DRIVER $I
done
coverage run $EXTRA ../../framework/TestDistributions.py

# code interface tests START
cd CodeInterfaceTests
coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_relap5_code_interface.xml interfaceCheck
coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_perturb_mammoth_bison_relap7.xml interfaceCheck
coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_mammoth_r7_bison_no_exe_hdf5_restart.xml interfaceCheck
coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_perturb_all_rattlesnake_bison.xml interfaceCheck
coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_perturb_mammoth_rattlesnake_bison.xml interfaceCheck
coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_relap5_code_interface_alias.xml interfaceCheck
coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_relap5_code_interface_multideck.xml interfaceCheck
coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_relap5_code_interface_multideck_choosing_deck_output.xml interfaceCheck
coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_OpenModelica_code_interface.xml interfaceCheck
coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_Dymola_code_interface.xml interfaceCheck
coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_Dymola_code_interface_timedep.xml interfaceCheck
coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_maap5_code_interface_forward.xml interfaceCheck
coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_maap5_code_interface_det.xml interfaceCheck
coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_maap5_code_interface_hybrid_det.xml interfaceCheck
coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_maap5_code_interface_adaptive_det.xml interfaceCheck
coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_maap5_code_interface_adaptive_hybrid_det.xml interfaceCheck
coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_maap5_code_interface_det_multibranch.xml interfaceCheck
coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_relap5_code_inss.xml interfaceCheck
# code interface tests END

if which Xvfb
then
    Xvfb :8888 &
    xvfbPID=$!
    oldDisplay=$DISPLAY
    export DISPLAY=:8888
    cd ../PostProcessors/TopologicalPostProcessor
    coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_topology_ui.xml interactiveCheck
    cd ../DataMiningPostProcessor/Clustering/
    coverage run $EXTRA $FRAMEWORK_DIR/Driver.py hierarchical_ui.xml interactiveCheck
    kill -9 $xvfbPID
    export DISPLAY=$oldDisplay
else
    ## Try these tests anyway, we can get some coverage out of them even if the
    ## UI fails or is unavailable.
    cd ../PostProcessors/TopologicalPostProcessor
    coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_topology_ui.xml interactiveCheck
    cd ../DataMiningPostProcessor/Clustering/
    coverage run $EXTRA $FRAMEWORK_DIR/Driver.py hierarchical_ui.xml interactiveCheck
fi

## Go to the final directory and generate the html documents
cd $SCRIPT_DIR/tests/framework
pwd
coverage html

