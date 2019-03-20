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

EXTRA="--rcfile=$FRAMEWORK_DIR/../tests/framework/.coveragerc --source=$FRAMEWORK_DIR --parallel-mode --omit=$FRAMEWORK_DIR/contrib/*"
export COVERAGE_FILE=`pwd`/.coverage

coverage erase
($FRAMEWORK_DIR/../run_tests "$@" --python-command="coverage run $EXTRA " || echo run_test done but some tests failed)

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
cd $SCRIPT_DIR/tests/
pwd
rm -f .cov_dirs
for FILE in `find . -name '.coverage.*'`; do dirname $FILE; done | sort | uniq > .cov_dirs
coverage combine `cat .cov_dirs`
coverage html

