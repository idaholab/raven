#!/bin/bash
export CHECKOUT_DIR=${CHECKOUT_DIR:=$HOME/raven_framework_checkout}
export JOBS=${JOBS:=1}
./raven_framework_git.sh

cd $CHECKOUT_DIR/trunk/raven
#GIT_VERSION=`git log -1 --format="%H_%ad" --date=short`  #`git rev-parse HEAD`
GIT_VERSION=`git describe`  #`git rev-parse HEAD`
(cd $CHECKOUT_DIR &&
tar --exclude=.git -cvzf raven_framework_${GIT_VERSION}_source.tar.gz trunk)

OS_NAME=`lsb_release -i -s || echo unknown`

cd $CHECKOUT_DIR/trunk/raven
make framework_modules
./run_tests -j$JOBS --re=framework --skip-config-checks || exit
cd $CHECKOUT_DIR/trunk/raven/doc/user_manual && make -j$JOBS
cd $CHECKOUT_DIR
tar --exclude=.git -cvzf raven_framework_${GIT_VERSION}_${OS_NAME}.tar.gz trunk
