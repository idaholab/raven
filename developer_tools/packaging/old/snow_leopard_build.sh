export https_proxy=$http_proxy
export INSTALL_DIR=/opt/raven_libs
rm -Rvf $INSTALL_DIR
export CFLAGS=-mmacosx-version-min=10.6
export CXXFLAGS=-mmacosx-version-min=10.6
./extra_libs.sh
../../raven_libs_script.sh
mkdir -p $HOME/raven_libs/root/opt
mv $INSTALL_DIR $HOME/raven_libs/root/opt
mkdir -p $HOME/raven_libs/root/opt/packages/environments
PROFILE_FILE=$HOME/raven_libs/root/opt/packages/environments/raven_libs_profile
echo 'export PYTHONPATH=/opt/raven_libs/lib/python2.7/site-packages/:$PYTHONPATH' > $PROFILE_FILE
chmod +x $PROFILE_FILE
mkdir -p $HOME/raven_libs/scripts
echo '#!/bin/bash' > $HOME/raven_libs/scripts/preflight
echo 'rm -Rf /opt/raven_libs/' >> $HOME/raven_libs/scripts/preflight
if test -x /Applications/Utilities/PackageMaker.app/Contents/MacOS/PackageMaker;
then
    PACKAGEMAKER=/Applications/Utilities/PackageMaker.app/Contents/MacOS/PackageMaker 
else
    PACKAGEMAKER=/Developer/Applications/Utilities/PackageMaker.app/Contents/MacOS/PackageMaker
fi
$PACKAGEMAKER --root $HOME/raven_libs/root --id raven_libs --verbose --title raven_libs  --scripts $HOME/raven_libs/scripts
rm -Rf raven_libs.pkg
mv root.pkg raven_libs.pkg

#Create dmg file.
rm -f raven_libs_base.dmg raven_libs.dmg
hdiutil create -size 1500m -fs HFS+ -volname "Raven Libraries" raven_libs_base.dmg
hdiutil attach raven_libs_base.dmg
cp -a raven_libs.pkg /Volumes/Raven\ Libraries
hdiutil detach /Volumes/Raven\ Libraries/
hdiutil convert raven_libs_base.dmg -format UDZO -o raven_libs.dmg


