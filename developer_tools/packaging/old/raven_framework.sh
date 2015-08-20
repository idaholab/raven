#!/bin/bash
SCRIPT_DIRNAME=`dirname $0`
SCRIPT_DIR=`(cd $SCRIPT_DIRNAME; pwd)`
CHECKOUT_DIR=${CHECKOUT_DIR:=$HOME/raven_framework_checkout}
RAVEN_DIR=${RAVEN_DIR:=$SCRIPT_DIR/../..}
#DEFINE EXTRA_SVN_ARGS to do things like get a specific revision
mkdir -p $CHECKOUT_DIR
cd $CHECKOUT_DIR
rm -Rf trunk
svn checkout $EXTRA_SVN_ARGS --depth empty https://hpcsc.inl.gov/svn/herd/trunk
cd trunk
EXTRA_SVN_ARGS=${EXTRA_SVN_ARGS:="-r `svnversion | sed 's/P$//'` "}
svn update $EXTRA_SVN_ARGS --set-depth empty raven
svn update $EXTRA_SVN_ARGS --set-depth infinity raven/framework
svn update $EXTRA_SVN_ARGS raven/README.txt
svn update $EXTRA_SVN_ARGS raven/Makefile
svn update $EXTRA_SVN_ARGS raven/raven.mk
svn update $EXTRA_SVN_ARGS raven/raven_libs_script.sh
svn update $EXTRA_SVN_ARGS raven/py3_raven_libs_script.sh
svn update $EXTRA_SVN_ARGS raven/backend_raven_libs_script.sh
svn update $EXTRA_SVN_ARGS --set-depth empty raven/tests/
svn update $EXTRA_SVN_ARGS --set-depth infinity raven/tests/framework
svn update $EXTRA_SVN_ARGS raven/run_tests
svn update $EXTRA_SVN_ARGS --set-depth infinity raven/scripts
svn update $EXTRA_SVN_ARGS --set-depth empty moose
svn update $EXTRA_SVN_ARGS --set-depth empty moose/framework
svn update $EXTRA_SVN_ARGS --set-depth empty moose/framework/scripts
svn update $EXTRA_SVN_ARGS --set-depth infinity moose/framework/scripts/TestHarness
svn update $EXTRA_SVN_ARGS --set-depth files moose/framework/scripts
svn update $EXTRA_SVN_ARGS --set-depth files moose/framework/scripts/common/
svn update $EXTRA_SVN_ARGS --set-depth infinity moose/python
svn update $EXTRA_SVN_ARGS --set-depth infinity crow
svn update $EXTRA_SVN_ARGS moose/framework/build.mk
svn update $EXTRA_SVN_ARGS moose/framework/moose.mk
svn update $EXTRA_SVN_ARGS --set-depth empty raven/doc
svn update $EXTRA_SVN_ARGS --set-depth infinity raven/doc/user_manual

#mkdir -p moose/libmesh/installed/bin
#cp $RAVEN_DIR/../moose/libmesh/installed/bin/libmesh-config  moose/libmesh/installed/bin
#mkdir -p moose/libmesh/installed/include/libmesh/
#cp $RAVEN_DIR/../moose/libmesh/installed/include/libmesh/libmesh_config.h moose/libmesh/installed/include/libmesh
#mkdir -p moose/libmesh/installed/contrib/bin
#cp $RAVEN_DIR/../moose/libmesh/installed/contrib/bin/libtool moose/libmesh/installed/contrib/bin
#cp $RAVEN_DIR/RAVEN-$METHOD raven
