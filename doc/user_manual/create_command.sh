#!/bin/bash

python ../../scripts/TestHarness/testers/RavenUtils.py --conda-create | fold -s -w60 | sed 's/$/\\/' > conda_command.txt
truncate --size=-2 conda_command.txt
