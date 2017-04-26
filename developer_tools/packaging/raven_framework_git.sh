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
git checkout $RAVEN_REV -- README.md
git checkout $RAVEN_REV -- NOTICE.txt
git checkout $RAVEN_REV -- LICENSE.txt
git checkout $RAVEN_REV -- Makefile
git checkout $RAVEN_REV -- *.mk
git checkout $RAVEN_REV -- developer_tools/py3_raven_libs_script.sh
git checkout $RAVEN_REV -- developer_tools/backend_raven_libs_script.sh
git checkout $RAVEN_REV -- developer_tools/createRegressionTestDocumentation.py
git checkout $RAVEN_REV -- tests/framework
git checkout $RAVEN_REV -- tests/cluster_tests
git checkout $RAVEN_REV -- run_tests
git checkout $RAVEN_REV -- backend_run_tests
git checkout $RAVEN_REV -- run_framework_tests
git checkout $RAVEN_REV -- build_framework
git checkout $RAVEN_REV -- scripts
git checkout $RAVEN_REV -- doc/user_manual
git checkout $RAVEN_REV -- doc/user_guide
git checkout $RAVEN_REV -- doc/make_docs.sh
git checkout $RAVEN_REV -- doc/qa_docs
git checkout $RAVEN_REV -- doc/tests
git checkout $RAVEN_REV -- doc/tex_inputs
git checkout $RAVEN_REV -- raven_framework
git checkout $RAVEN_REV -- src/contrib
git checkout $RAVEN_REV -- include/contrib
git checkout $RAVEN_REV -- setup.py
git checkout $RAVEN_REV -- setup3.py
git checkout $RAVEN_REV -- crow
git checkout $RAVEN_REV -- moose
git checkout $RAVEN_REV -- .gitmodules

MOOSE_REV=`git submodule status moose | tr -d - | cut -d ' ' -f 1`

#fix version tex file.
cd $MAIN_DIR/raven/doc/user_manual
make ../version.tex

cd $MAIN_DIR/raven/
echo RAVEN revision `git describe` > Version.txt
echo RAVEN id `git rev-parse HEAD` >> Version.txt
echo MOOSE id $MOOSE_REV >> Version.txt

cd $MAIN_DIR/raven/

git clone --shared --no-checkout $PULL_DIRECTORY/raven/moose
cd moose
git checkout $MOOSE_REV -- framework/scripts/find_dep_apps.py
git checkout $MOOSE_REV -- framework/Makefile #Can be removed in future
git checkout $MOOSE_REV -- python/path_tool.py
git checkout $MOOSE_REV -- python/FactorySystem
git checkout $MOOSE_REV -- python/run_tests
git checkout $MOOSE_REV -- python/mooseutils/*.py
git checkout $MOOSE_REV -- python/TestHarness
git checkout $MOOSE_REV -- python/argparse
git checkout $MOOSE_REV -- COPYRIGHT
git checkout $MOOSE_REV -- COPYING
git checkout $MOOSE_REV -- LICENSE
git checkout $MOOSE_REV -- README.md
# Remove these because run_tests finds TestHarness tests and they fail
# without the rest of moose
rm -Rf python/TestHarness/tests

echo using revisions MOOSE: $MOOSE_REV RAVEN: $RAVEN_REV
