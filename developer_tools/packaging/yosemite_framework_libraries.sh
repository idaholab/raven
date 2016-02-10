export https_proxy=$http_proxy
export INSTALL_DIR=/opt/raven_libs

rm -Rvf $INSTALL_DIR
../backend_raven_libs_script.sh
ls $INSTALL_DIR/bin
mkdir -p $HOME/raven_libs/root/opt
mv $INSTALL_DIR $HOME/raven_libs/root/opt
mkdir -p $HOME/raven_libs/root/opt/raven_libs/environments
PROFILE_FILE=$HOME/raven_libs/root/opt/raven_libs/environments/raven_libs_profile
cat - > $PROFILE_FILE << RAVEN_PROFILE
export PYTHONPATH="/opt/raven_libs/lib/python2.7/site-packages/:\$PYTHONPATH"
export PATH="\$PATH:/opt/raven_libs/bin"
RAVEN_PROFILE

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
POSTINSTALL

chmod +x $HOME/raven_libs/scripts/postinstall

rm -Rf raven_libs.pkg
pkgbuild --root $HOME/raven_libs/root --identifier raven_libs  --scripts $HOME/raven_libs/scripts raven_libs.pkg

#Create dmg file.
rm -f raven_libs_base.dmg raven_libs.dmg
hdiutil create -size 200m -fs HFS+ -volname "Raven Libraries" raven_libs_base.dmg
hdiutil attach raven_libs_base.dmg
cp -a raven_libs.pkg /Volumes/Raven\ Libraries
hdiutil detach /Volumes/Raven\ Libraries/
hdiutil convert raven_libs_base.dmg -format UDZO -o raven_libs.dmg


