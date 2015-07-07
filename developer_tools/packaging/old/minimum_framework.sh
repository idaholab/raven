#!/bin/bash
SCRIPT_DIRNAME=`dirname $0`
SCRIPT_DIR=`(cd $SCRIPT_DIRNAME; pwd)`
CHECKOUT_DIR=${CHECKOUT_DIR:=$HOME/raven_framework_checkout}

#This script is if crow is already compiled such as with:
#python setup.py install --user
#DEFINE EXTRA_SVN_ARGS to do things like get a specific revision
mkdir -p $CHECKOUT_DIR
cd $CHECKOUT_DIR
rm -Rf trunk
svn checkout $EXTRA_SVN_ARGS --depth empty https://hpcsc.inl.gov/svn/herd/trunk
cd trunk
svn update $EXTRA_SVN_ARGS --set-depth empty raven
svn update $EXTRA_SVN_ARGS raven/README.txt
svn update $EXTRA_SVN_ARGS --set-depth infinity raven/framework
svn update $EXTRA_SVN_ARGS --set-depth empty raven/tests/
svn update $EXTRA_SVN_ARGS --set-depth infinity raven/tests/framework
svn update $EXTRA_SVN_ARGS raven/run_tests
svn update $EXTRA_SVN_ARGS raven/raven_framework
svn update $EXTRA_SVN_ARGS --set-depth empty moose
svn update $EXTRA_SVN_ARGS --set-depth infinity moose/python
svn update $EXTRA_SVN_ARGS --set-depth empty moose/framework
svn update $EXTRA_SVN_ARGS --set-depth files moose/framework/scripts
svn update $EXTRA_SVN_ARGS --set-depth empty raven/scripts
svn update $EXTRA_SVN_ARGS --set-depth infinity raven/scripts/TestHarness
svn update $EXTRA_SVN_ARGS --set-depth empty raven/doc
svn update $EXTRA_SVN_ARGS --set-depth infinity raven/doc/user_manual


