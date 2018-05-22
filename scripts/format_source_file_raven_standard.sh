#!/bin/bash
SCRIPT_NAME=`readlink $0`
if test -x "$SCRIPT_NAME";
then
    SCRIPT_DIRNAME=`dirname $SCRIPT_NAME`
else
    SCRIPT_DIRNAME=`dirname $0`
fi
SCRIPT_DIR=`(cd $SCRIPT_DIRNAME; pwd)`

echo FIXING  docstrings of: $1 && python -O $SCRIPT_DIR/fix_raven_docstrings.py "$1" "-i" "-c_max" "119"
echo FIXING source code of: $1 && yapf --style='{based_on_style: pep8, indent_width: 2, 119}' -i $1
bash $SCRIPT_DIR/../developer_tools/delete_trailing_whitespace.sh "$1"

echo COMPLETE!
