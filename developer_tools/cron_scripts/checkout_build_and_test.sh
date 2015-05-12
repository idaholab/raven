#!/bin/bash

PROJECT_DIR=${PROJECT_DIR:=$HOME/test_projects}
test -d $PROJECT_DIR && (cd $PROJECT_DIR && rm -Rvf moose relap-7 raven crow)
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

git clone https://github.com/idaholab/moose.git

git clone git@hpcgitlab.inl.gov:idaholab/relap-7.git

cd relap-7
git submodule update --init contrib/iapws

cd ..

git clone git@hpcgitlab.inl.gov:idaholab/crow.git
git clone git@hpcgitlab.inl.gov:idaholab/raven.git

cd raven
make -j3

#If make failed the first time, try again
if test $? -ne 0;
then
    (cd ../crow && git clean -f -x -d .)
    git clean -f -x -d .
    make -j3 || exit
fi

./run_tests -j3 --no-color

cd inputs/mpi_driver_test/
./test_qsubs.sh

#If test qsubs failed the first time, try again
if test $? -ne 0;
then
    ./test_qsubs.sh
fi
