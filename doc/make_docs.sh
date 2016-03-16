#!/bin/bash

SCRIPT_NAME=`readlink $0`
if test -x "$SCRIPT_NAME";
then
    SCRIPT_DIRNAME=`dirname $SCRIPT_NAME`
else
    SCRIPT_DIRNAME=`dirname $0`
fi
SCRIPT_DIR=`(cd $SCRIPT_DIRNAME; pwd)`
cd $SCRIPT_DIR

for DIR in user_manual qa_docs tests; do
    cd $DIR
    if make; then
        echo Successfully made docs in $DIR
    else
        echo Failed to make docs in $DIR
        exit -1
    fi
    cd $SCRIPT_DIR
done

rm -Rvf pdfs
mkdir pdfs
for DOC in qa_docs/raven_sdd.pdf qa_docs/test_plan.pdf qa_docs/requirements.pdf  user_manual/raven_user_manual.pdf tests/analytic_tests.pdf; do
    cp $DOC pdfs/
done


