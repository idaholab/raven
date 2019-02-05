#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
xmllint --noout --schema $DIR/XSDSchemas/raven.xsd $1
