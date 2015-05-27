#!/bin/bash
cd `dirname $0`
./developer_tools/backend_raven_libs_script.sh

if test ""`which python3-config` != "" -a ""$CROW_USE_PYTHON3 == TRUE
then
    if test "`uname -sr | sed 's/\..*//'`" = "Darwin 13"
    then
	unset PYTHONPATH
    fi
    ./developer_tools/py3_raven_libs_script.sh
fi
