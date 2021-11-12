#!/bin/bash

cat > pip_commands.txt <<EOF
#Create a new virtual environment in directory raven_libs
virtualenv raven_libs
#Use the virtual environment
source raven_libs/bin/activate
#Use pip to install needed libraries
EOF
python ../../scripts/library_handler.py pip --action install | fold -s -w60 | sed 's/$/\\/' | python -c 'import sys; sys.stdout.write(sys.stdin.read()[:-2])'>> pip_commands.txt

