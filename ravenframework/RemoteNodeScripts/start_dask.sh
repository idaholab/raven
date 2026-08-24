#Created on Mar 21, 2023
#
# OUTPUT FILE FOR LOGGING
OUTFILE=$1
SCHEDULER_FILE=$2
NUM_CPUS=$3
RAVEN_FRAMEWORK_DIR=$4
REMOTE_BASH=$5
WORKING_DIR=$6
PYTHONPATH=$7
export PYTHONPATH

echo starting >> "$OUTFILE"

cd "$WORKING_DIR" || { echo "cannot cd to $WORKING_DIR" >> "$OUTFILE"; exit 1; }
# the remote bash profile is optional
if [ -n "$REMOTE_BASH" ] && [ -f "$REMOTE_BASH" ]
  then
  source "$REMOTE_BASH" >> "$OUTFILE" 2>&1
fi

which dask >> "$OUTFILE" 2>&1
hostname >> "$OUTFILE"
echo PYTHONPATH "$PYTHONPATH" >> "$OUTFILE"

dask worker --nworkers "$NUM_CPUS" --scheduler-file "$SCHEDULER_FILE" >> "$OUTFILE" 2>&1 &
