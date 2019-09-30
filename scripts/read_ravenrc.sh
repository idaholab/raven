#!/bin/bash

# USES:
# read .ravenrc file

ECE_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RAVEN_UTILS=${ECE_SCRIPT_DIR}/TestHarness/testers/RavenUtils.py

# fail if ANYTHING this script fails (mostly, there are exceptions)
set -e

function read_ravenrc ()
{
  # $1 should be the keyword we're looking for
  # returns keyword argument through echo
  ## note that "| xargs" trims leading and trailing whitespace
  local TARGET=`echo $1 | xargs`
  # location of the RC file
  local RCNAME="${ECE_SCRIPT_DIR}/../.ravenrc"
  # if the RC file exists, loop through it and read keyword arguments split by "="
  if [ -f "$RCNAME" ]; then
    while IFS='=' read -r KEY ARG || [[ -n "$keyarg" ]]; do
      # trim whitespace
      KEY=`echo $KEY | xargs`
      ARG=`echo $ARG | xargs`
      # check for key match
      if [ "$KEY" = "$TARGET" ]; then
        echo "$ARG"
        return 0
      fi
    done < ${RCNAME}
  fi
  # if not found, return empty
  echo ''
}

