#!/bin/bash
PYTHON_CMD=${PYTHON_CMD:=python}
SCRIPT_DIRNAME=`dirname $0`
SCRIPT_DIR=`(cd $SCRIPT_DIRNAME; pwd)`
# load raven libraries
. scripts/establish_conda_env.sh --load

$PYTHON_CMD ${SCRIPT_DIR}/validate_xml.py

#  #BUILD_DIR=${BUILD_DIR:=$HOME/raven_libs/build}
#  #INSTALL_DIR=${INSTALL_DIR:=$HOME/raven_libs/pylibs}
#  PYTHON_CMD=${PYTHON_CMD:=python}
#  JOBS=${JOBS:=1}
#  SCRIPT_DIRNAME=`dirname $0`
#  SCRIPT_DIR=`(cd $SCRIPT_DIRNAME; pwd)`
#  
#  ORIGPYTHONPATH="$PYTHONPATH"
#  
#  maxlen=$(($(tput cols) < 126 ? $(tput cols) : 126))
#  
#  # Not Mac-compatible:
#  #TEST_DIRS="${SCRIPT_DIR}/../tests/cluster_tests "`find ${SCRIPT_DIR}/../tests/framework -name tests -printf "%h\n"`
#  TEST_FILES="${SCRIPT_DIR}/../tests/cluster_tests/tests "`find ${SCRIPT_DIR}/../tests/framework -name tests`
#  TEST_DIRS=""
#  for file in ${TEST_FILES}
#  do
#      TEST_DIRS+=" $(dirname $file)"
#  done
#  
#  failed_tests=0
#  passed_tests=0
#  
#  CONVERSION_SCRIPT_DIR="${SCRIPT_DIR}/../scripts/conversionScripts"
#  
#  # get_coverage_tests.py now gets all the tests in all folders under raven/tests
#  # for testdir in ${TEST_DIRS}
#  # do
#  #     echo -e "\033[1;32mValidating $testdir "
#  #    cd $testdir
#      echo -e "\033[1;32mValidating tests... "
#      for I in $(python ${SCRIPT_DIR}/get_coverage_tests.py --skip-fails)
#      do
#          # convert the input files contain the external xml to normal raven input files and then validate
#          $PYTHON_CMD $CONVERSION_SCRIPT_DIR/externalXMLNode.py $I > /dev/null
#          # echo -e "\033[1;32mThe following script: $I has been converted for xsd validate"
#          echo -en "\033[0mValidating $I"
#          startlen=$((11+${#I}))
#          VALOUT=$(xmllint --noout --schema  ${SCRIPT_DIR}/XSDSchemas/raven.xsd $I 2>&1)
#          if test $? -eq 0;
#          then
#              periodlength=$((maxlen - startlen - 11))
#              printf '%0.s.' $(seq 1 $periodlength)
#              echo -e "\033[1;32m validated!"
#              passed_tests=$(($passed_tests + 1))
#          else
#              periodlength=$((maxlen - startlen - 8))
#              printf '%0.s.' $(seq 1 $periodlength)
#              echo -e "\033[1;31m FAILED!"
#              echo -e "$VALOUT"
#              failed_tests=$(($failed_tests + 1))
#          fi
#          # Move the file back to its original state
#          mv $I.bak $I
#          # echo -e "\033[1;32m$I has been converted back!"
#      done
#  #    echo -e "\033[0m--------------------------------------------------"
#  # done
#  echo -e "\033[1;32mPassed $passed_tests \033[1;31mFailed $failed_tests\033[0m"
#  
#  # test_External_XML.xml should transformed back, this is the test that test the external xml input functionality
#  # mv ${SCRIPT_DIR}/../tests/framework/test_External_XML.xml.bak ${SCRIPT_DIR}/../tests/framework/test_External_XML.xml
#  # echo -e "\033[1;32m test_External_XML.xml has been converted back!"
#  
#  exit $failed_tests
