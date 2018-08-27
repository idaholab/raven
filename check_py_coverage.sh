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
    #SHA256=56e448f051a201c5ebbaa86a5efd0ca90d327204d8b059ab25ad0f35fbfd79f1
    $DOWNLOADER https://files.pythonhosted.org/packages/35/fe/e7df7289d717426093c68d156e0fd9117c8f4872b6588e8a8928a0f68424/coverage-4.5.1.tar.gz
    tar -xvzf coverage-3.7.1.tar.gz
    cd coverage-3.7.1
    (unset CC CXX; $PYTHON_CMD setup.py install --prefix=$INSTALL_DIR)
fi

update_python_path

cd $SCRIPT_DIR

cd tests/framework
#coverage help run
FRAMEWORK_DIR=`(cd ../../framework && pwd)`

source $SCRIPT_DIR/scripts/establish_conda_env.sh --quiet --load
# get display var
DISPLAY_VAR=`(echo $DISPLAY)`
# reset it
export DISPLAY=

EXTRA="--rcfile=$FRAMEWORK_DIR/../tests/framework/.coveragerc --source=$FRAMEWORK_DIR -a --omit=$FRAMEWORK_DIR/contrib/*"
export COVERAGE_FILE=`pwd`/.coverage

coverage erase
#skip test_rom_trainer.xml
DRIVER=$FRAMEWORK_DIR/Driver.py
echo ...Running Code Interface tests...
# get the tests runnable by RAVEN (interface check)
for I in $(python ${SCRIPT_DIR}/developer_tools/get_coverage_tests.py --get-interface-check-tests --skip-fails)
do
    DIR=`dirname $I`
    BASE=`basename $I`
    #echo Running $DIR $BASE
    cd $DIR
    echo coverage run $EXTRA $DRIVER $I interfaceCheck
    coverage run $EXTRA $DRIVER $I interfaceCheck
done

echo ...Running Unit tests...
# get the tests runnable by RAVEN (python tests (unit-tests))
for I in $(python ${SCRIPT_DIR}/developer_tools/get_coverage_tests.py --get-python-tests --skip-fails)
do
    DIR=`dirname $I`
    BASE=`basename $I`
    #echo Running $DIR $BASE
    cd $DIR
    echo coverage run $EXTRA $I
    coverage run $EXTRA $I
done

echo ...Running Verification tests...
# get the tests runnable by RAVEN (not interface check)
for I in $(python ${SCRIPT_DIR}/developer_tools/get_coverage_tests.py)
do
    DIR=`dirname $I`
    BASE=`basename $I`
    #echo Running $DIR $BASE
    if [ -d "$DIR" ]; then
        cd $DIR
        echo coverage run $EXTRA $DRIVER $I
        coverage run $EXTRA $DRIVER $I || true
    fi
done

#get DISPLAY BACK
DISPLAY=$DISPLAY_VAR

if which Xvfb
then
    Xvfb :8888 &
    xvfbPID=$!
    oldDisplay=$DISPLAY
    export DISPLAY=:8888
    cd $FRAMEWORK_DIR/../tests/framework/PostProcessors/TopologicalPostProcessor
    coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_topology_ui.xml interactiveCheck || true
    cd $FRAMEWORK_DIR/../tests/framework/PostProcessors/DataMiningPostProcessor/Clustering/
    coverage run $EXTRA $FRAMEWORK_DIR/Driver.py hierarchical_ui.xml interactiveCheck || true
    kill -9 $xvfbPID || true
    export DISPLAY=$oldDisplay
else
    ## Try these tests anyway, we can get some coverage out of them even if the
    ## UI fails or is unavailable.
    cd $FRAMEWORK_DIR/../tests/framework/PostProcessors/TopologicalPostProcessor
    coverage run $EXTRA $FRAMEWORK_DIR/Driver.py test_topology_ui.xml interactiveCheck || true
    cd $FRAMEWORK_DIR/../tests/framework/PostProcessors/DataMiningPostProcessor/Clustering/
    coverage run $EXTRA $FRAMEWORK_DIR/Driver.py hierarchical_ui.xml interactiveCheck || true
fi

## Go to the final directory and generate the html documents
cd $SCRIPT_DIR/tests/framework
pwd
coverage html

