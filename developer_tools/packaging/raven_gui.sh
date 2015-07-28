#!/bin/bash
SCRIPT_DIRNAME=`dirname $0`
SCRIPT_DIR=`(cd $SCRIPT_DIRNAME; pwd)`
CHECKOUT_DIR=${CHECKOUT_DIR:=$HOME/raven_gui_checkout}
RAVEN_DIR=${RAVEN_DIR:=$SCRIPT_DIR/../..}
mkdir $CHECKOUT_DIR
cd $CHECKOUT_DIR
svn checkout --depth empty https://hpcsc.inl.gov/svn/herd/trunk
cd trunk
svn update --set-depth empty moose
svn update --set-depth infinity moose/gui
svn update --set-depth empty raven
svn update --set-depth infinity raven/gui
svn update --set-depth empty r7_moose
svn update --set-depth infinity r7_moose/gui
svn update --set-depth empty relap-7
svn update --set-depth infinity relap-7/gui
svn update --set-depth infinity moose/python
cp $RAVEN_DIR/syntax_dump_RAVEN-$METHOD $RAVEN_DIR/yaml_dump_RAVEN-$METHOD $RAVEN_DIR/yaml_dump_RAVEN-${METHOD}_raw raven/
cd $CHECKOUT_DIR
rm -f raven_gui.zip
zip raven_gui.zip -r trunk -x '*/.svn/*'
