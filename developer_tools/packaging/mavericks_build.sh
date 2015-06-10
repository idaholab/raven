export https_proxy=$http_proxy
export INSTALL_DIR=/opt/raven_libs
source /opt/moose/environments/moose_profile
rm -Rvf $INSTALL_DIR
../../backend_raven_libs_script.sh
mkdir -p $HOME/raven_libs/root/opt
mv $INSTALL_DIR $HOME/raven_libs/root/opt
mkdir -p $HOME/raven_libs/root/opt/raven_libs/environments
PROFILE_FILE=$HOME/raven_libs/root/opt/raven_libs/environments/raven_libs_profile
echo 'export PYTHONPATH=/opt/raven_libs/lib/python2.7/site-packages/:$PYTHONPATH' > $PROFILE_FILE
chmod +x $PROFILE_FILE
mkdir -p $HOME/raven_libs/scripts
cat - > $HOME/raven_libs/scripts/preflight <<PREFLIGHT
#!/bin/bash
rm -Rf /opt/raven_libs/
PREFLIGHT
chmod +x $HOME/raven_libs/scripts/preflight

cat - > $HOME/raven_libs/scripts/postinstall <<POSTINSTALL
#!/bin/bash
echo Running Raven libs postinstall 
echo HOME = \$HOME
if grep '. /opt/raven_libs/environments/raven_libs_profile'  \$HOME/.bash_profile; then echo Already sourcing /opt/raven_libs/environments/raven_libs_profile; else  
cat - >> \$HOME/.bash_profile <<EOF
#source raven libs environment
if [ -f /opt/raven_libs/environments/raven_libs_profile ]; then
        . /opt/raven_libs/environments/raven_libs_profile 
fi
EOF
fi

if which python3-config; then echo Python3 already installed; else
installer -pkg /Volumes/Raven\ Libraries/Python.mpkg -target /
fi
POSTINSTALL

chmod +x $HOME/raven_libs/scripts/postinstall

rm -Rf raven_libs.pkg
pkgbuild --root $HOME/raven_libs/root --identifier raven_libs  --scripts $HOME/raven_libs/scripts raven_libs.pkg

#Get Python 
curl -C - -L -O http://www.python.org/ftp/python/3.3.5/python-3.3.5-macosx10.6.dmg
hdiutil attach python-3.3.5-macosx10.6.dmg 
cp -a /Volumes/Python\ 3.3.5/Python.mpkg .
hdiutil detach /Volumes/Python\ 3.3.5/


#Create dmg file.
rm -f raven_libs_base.dmg raven_libs.dmg
hdiutil create -size 200m -fs HFS+ -volname "Raven Libraries" raven_libs_base.dmg
hdiutil attach raven_libs_base.dmg
cp -a raven_libs.pkg /Volumes/Raven\ Libraries
cp -a Python.mpkg /Volumes/Raven\ Libraries
hdiutil detach /Volumes/Raven\ Libraries/
hdiutil convert raven_libs_base.dmg -format UDZO -o raven_libs.dmg


