#!/bin/bash
SCRIPT_DIRNAME=`dirname $0`
SCRIPT_DIR=`(cd $SCRIPT_DIRNAME; pwd)`
CHECKOUT_DIR=${CHECKOUT_DIR:=$HOME/raven_framework_checkout_git}

#This script is if crow is already compiled such as with:
#python setup.py install --user

RAVEN_DIR=${RAVEN_DIR:=$SCRIPT_DIR/../..}
PROJECT_DIR=$RAVEN_DIR/..
MAIN_DIR=$CHECKOUT_DIR/trunk
mkdir -p $MAIN_DIR
cd $MAIN_DIR
RAVEN_BRANCH=${RAVEN_BRANCH:=master}
MOOSE_BRANCH=${MOOSE_BRANCH:=master}
echo using branches MOOSE: $MOOSE_BRANCH RAVEN: $RAVEN_BRANCH

rm -Rf moose raven
git clone --branch $MOOSE_BRANCH --shared --no-checkout $PROJECT_DIR/moose
git clone --branch $RAVEN_BRANCH --shared --no-checkout $PROJECT_DIR/raven

cd $MAIN_DIR/raven
git checkout $RAVEN_BRANCH -- framework
git checkout $RAVEN_BRANCH -- README.txt
git checkout $RAVEN_BRANCH -- tests/framework
git checkout $RAVEN_BRANCH -- run_tests
git checkout $RAVEN_BRANCH -- run_framework_tests
git checkout $RAVEN_BRANCH -- scripts/TestHarness
git checkout $RAVEN_BRANCH -- doc/user_manual
git checkout $RAVEN_BRANCH -- raven_framework

cd $MAIN_DIR/moose
git checkout $MOOSE_BRANCH -- framework/scripts
git checkout $MOOSE_BRANCH -- python
