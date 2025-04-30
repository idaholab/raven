#!/bin/bash

version=$(cat version.txt)
echo "Updating to version "$version
rm -rf build/*
rm -rf dist/*
python setup.py bdist_wheel --universal
#twine register dist/ndtamr-$version-py2.py3-none-any.whl
twine upload dist/* 
