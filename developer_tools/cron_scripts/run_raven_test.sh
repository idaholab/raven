#!/bin/bash

#Change these to whatever is correct locally
RAVEN_HOME=$HOME/raven/falcon/fal_pack/
RESULT_EMAIL=joshua.cogliati@inl.gov

DATE_STRING=`date '+%F_%R'`
echo "Started" > $HOME/raven/logs/cron_started_${DATE_STRING}

#XXX some versions of bash have the problem that if output is redirected,
# sourcing and eval do not work.

source /etc/profile #2>&1 | tee -a $HOME/raven/logs/cron_output_${DATE_STRING}

eval `/usr/bin/modulecmd bash load pbs raven-devel-gcc` #2>&1 | tee -a $HOME/raven/logs/cron_output_${DATE_STRING}

env 2>&1 | tee -a $HOME/raven/logs/cron_output_${DATE_STRING}

$RAVEN_HOME/raven/developer_tools/cron_scripts/checkout_build_and_test.sh 2>&1 | tee -a $HOME/raven/logs/cron_output_${DATE_STRING}

tail -n 500 $HOME/raven/logs/cron_output_${DATE_STRING} | mailx -s "RAVEN cron output ${DATE_STRING}" $RESULT_EMAIL
