#!/bin/bash
SCRIPT_DIRNAME=`dirname $0`
SCRIPT_DIR=`(cd $SCRIPT_DIRNAME; pwd)`
CHECKOUT_DIR=${CHECKOUT_DIR:=$HOME/raven_framework_checkout_git}
RAVEN_DIR=${RAVEN_DIR:=$SCRIPT_DIR/../..}
PULL_DIRECTORY=$RAVEN_DIR/..
MAIN_DIR=$CHECKOUT_DIR/
mkdir -p $MAIN_DIR
cd $MAIN_DIR
RAVEN_BRANCH=${RAVEN_BRANCH:=master}
RAVEN_REV=${RAVEN_REV:=$RAVEN_BRANCH}

rm -Rf raven
git clone --branch $RAVEN_BRANCH --shared --no-checkout $PULL_DIRECTORY/raven
cd raven

cd $MAIN_DIR/raven
git checkout $RAVEN_REV -- framework
git checkout $RAVEN_REV -- README.txt
git checkout $RAVEN_REV -- Makefile
git checkout $RAVEN_REV -- *.mk
git checkout $RAVEN_REV -- raven_libs_script.sh
git checkout $RAVEN_REV -- developer_tools/py3_raven_libs_script.sh
git checkout $RAVEN_REV -- developer_tools/backend_raven_libs_script.sh
git checkout $RAVEN_REV -- tests/framework
git checkout $RAVEN_REV -- tests/cluster_tests
git checkout $RAVEN_REV -- run_tests
git checkout $RAVEN_REV -- run_framework_tests
git checkout $RAVEN_REV -- scripts
git checkout $RAVEN_REV -- doc/user_manual
git checkout $RAVEN_REV -- raven_framework
git checkout $RAVEN_REV -- src/contrib
git checkout $RAVEN_REV -- include/contrib
git checkout $RAVEN_REV -- setup.py
git checkout $RAVEN_REV -- setup3.py
git checkout $RAVEN_REV -- crow
git checkout $RAVEN_REV -- moose
git checkout $RAVEN_REV -- .gitmodules

CROW_REV=`git submodule status crow | tr -d - | cut -d ' ' -f 1`
MOOSE_REV=`git submodule status moose | tr -d - | cut -d ' ' -f 1`

cd $MAIN_DIR/raven/
git clone --shared --no-checkout $PULL_DIRECTORY/crow
cd crow
git checkout $CROW_REV -- .

cd $MAIN_DIR/raven/
# Copy these because run_tests sometimes finds the crow tests.
cp crow/scripts/TestHarness/testers/* scripts/TestHarness/testers/

git clone --shared --no-checkout $PULL_DIRECTORY/moose
cd moose
git checkout $MOOSE_REV -- framework/scripts
git checkout $MOOSE_REV -- framework/Makefile
git checkout $MOOSE_REV -- python
git checkout $MOOSE_REV -- framework/build.mk
git checkout $MOOSE_REV -- framework/moose.mk
git checkout $MOOSE_REV -- framework/app.mk
git checkout $MOOSE_REV -- modules/modules.mk
git checkout $MOOSE_REV -- COPYRIGHT
git checkout $MOOSE_REV -- COPYING
git checkout $MOOSE_REV -- EXPORT_CONTROL
git checkout $MOOSE_REV -- LICENSE
git checkout $MOOSE_REV -- README.md

echo using revisions MOOSE: $MOOSE_REV RAVEN: $RAVEN_REV CROW: $CROW_REV
