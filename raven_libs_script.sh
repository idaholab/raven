#!/bin/bash
cd `dirname $0`
./backend_raven_libs_script.sh

if test ""`which python3-config` != ""
then
    ./py3_raven_libs_script.sh
fi
