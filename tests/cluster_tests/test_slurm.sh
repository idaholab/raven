#!/bin/bash

#This script is used on the INL slurm cluster machine to test
# the cluster interface.

pushd ../../ravenframework
RAVEN_FRAMEWORK_DIR=$(pwd)
popd

echo Current directory: `pwd`

cd ${RAVEN_FRAMEWORK_DIR}/..
# TODO running plugin qsub tests?
./run_tests -j3 --raven --only-run-types="slurm" --re=cluster_tests
EXIT=$?

exit $EXIT
