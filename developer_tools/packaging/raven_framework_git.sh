#!/bin/bash
SCRIPT_DIRNAME=`dirname $0`
SCRIPT_DIR=`(cd $SCRIPT_DIRNAME; pwd)`
CHECKOUT_DIR=${CHECKOUT_DIR:=$HOME/raven_framework_checkout_git}
RAVEN_DIR=${RAVEN_DIR:=$SCRIPT_DIR/../..}
PROJECT_DIR=$RAVEN_DIR/..
MAIN_DIR=$CHECKOUT_DIR/trunk
mkdir -p $MAIN_DIR
cd $MAIN_DIR
RAVEN_BRANCH=${RAVEN_BRANCH:=master}
MOOSE_BRANCH=${MOOSE_BRANCH:=master}
echo using branches MOOSE: $MOOSE_BRANCH RAVEN: $RAVEN_BRANCH

rm -Rf moose crow raven
git clone --branch $MOOSE_BRANCH --shared --no-checkout $PROJECT_DIR/moose
git clone --branch $RAVEN_BRANCH --shared --no-checkout $PROJECT_DIR/crow
git clone --branch $RAVEN_BRANCH --shared --no-checkout $PROJECT_DIR/raven

cd $MAIN_DIR/raven
git checkout $RAVEN_BRANCH -- framework
git checkout $RAVEN_BRANCH -- README.txt
git checkout $RAVEN_BRANCH -- Makefile
git checkout $RAVEN_BRANCH -- *.mk
git checkout $RAVEN_BRANCH -- raven_libs_script.sh
git checkout $RAVEN_BRANCH -- developer_tools/py3_raven_libs_script.sh
git checkout $RAVEN_BRANCH -- developer_tools/backend_raven_libs_script.sh
git checkout $RAVEN_BRANCH -- tests/framework
git checkout $RAVEN_BRANCH -- run_tests
git checkout $RAVEN_BRANCH -- run_framework_tests
git checkout $RAVEN_BRANCH -- scripts
git checkout $RAVEN_BRANCH -- doc/user_manual
git checkout $RAVEN_BRANCH -- raven_framework
git checkout $RAVEN_BRANCH -- src/contrib
git checkout $RAVEN_BRANCH -- include/contrib
git checkout $RAVEN_BRANCH -- setup.py
git checkout $RAVEN_BRANCH -- setup3.py

cd $MAIN_DIR/crow
git checkout $RAVEN_BRANCH -- .

cd $MAIN_DIR/moose
git checkout $MOOSE_BRANCH -- framework/scripts
git checkout $MOOSE_BRANCH -- framework/Makefile
git checkout $MOOSE_BRANCH -- python
git checkout $MOOSE_BRANCH -- framework/build.mk
git checkout $MOOSE_BRANCH -- framework/moose.mk
git checkout $MOOSE_BRANCH -- framework/app.mk
git checkout $MOOSE_BRANCH -- modules/modules.mk
