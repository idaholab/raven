#!/bin/bash

python ../../scripts/library_handler.py conda --action create | fold -s -w60 | sed 's/$/\\/' | python -c 'import sys; sys.stdout.write(sys.stdin.read()[:-2])'> conda_command.txt

