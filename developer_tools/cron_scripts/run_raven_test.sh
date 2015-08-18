#!/bin/bash

SCRIPT_NAME=`readlink $0`
if test -x "$SCRIPT_NAME";
then
    SCRIPT_DIRNAME=`dirname $SCRIPT_NAME`
else
    SCRIPT_DIRNAME=`dirname $0`
fi
SCRIPT_DIR=`(cd $SCRIPT_DIRNAME; pwd)`

#Change these to whatever is correct locally
# where to run the initial script from
RAVEN_HOME=`(cd ${SCRIPT_DIR}/../../../ && pwd)`
#Where the email should say it is from
FROM_EMAIL=${FROM_EMAIL:=`git config user.email`}
#Where to email the result to
RESULT_EMAIL=${RESULT_EMAIL:=$FROM_EMAIL}
#Where the files will be checked out.
export PROJECT_DIR=${PROJECT_DIR:=$HOME/test_projects}

DATE_STRING=`date '+%F_%R'`
echo "Started" > $HOME/raven/logs/cron_started_${DATE_STRING}

#XXX some versions of bash have the problem that if output is redirected,
# sourcing and eval do not work.

source /etc/profile #2>&1 | tee -a $HOME/raven/logs/cron_output_${DATE_STRING}

#eval `/usr/bin/modulecmd bash load pbs raven-devel-gcc` #2>&1 | tee -a $HOME/raven/logs/cron_output_${DATE_STRING}
module load pbs raven-devel-gcc

env 2>&1 | tee -a $HOME/raven/logs/cron_output_${DATE_STRING}

$RAVEN_HOME/raven/developer_tools/cron_scripts/checkout_build_and_test.sh 2>&1 | tee -a $HOME/raven/logs/cron_output_${DATE_STRING}

SHORT_OUTPUT=$HOME/raven/logs/cron_output_short_${DATE_STRING}
echo Raven version: > $SHORT_OUTPUT
(cd ${PROJECT_DIR}/raven && git --no-pager log -1) >> $SHORT_OUTPUT
echo >> $SHORT_OUTPUT
echo Crow version: >> $SHORT_OUTPUT
(cd ${PROJECT_DIR}/crow && git --no-pager log -1) >> $SHORT_OUTPUT
echo >> $SHORT_OUTPUT
tail -n 500 $HOME/raven/logs/cron_output_${DATE_STRING} >> $SHORT_OUTPUT
cat $SHORT_OUTPUT | mailx -r $FROM_EMAIL -s "RAVEN cron output ${DATE_STRING}" $RESULT_EMAIL
