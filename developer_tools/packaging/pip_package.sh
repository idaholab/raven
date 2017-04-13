#!/bin/bash
export https_proxy=$http_proxy
export INSTALL_DIR=$HOME/raven_libs/root/opt/raven_libs

rm -Rvf $INSTALL_DIR
./pip_ve_osx.sh

#Create raven environment script.
mkdir -p $HOME/raven_libs/root/opt/raven_libs/environments
PROFILE_FILE=$HOME/raven_libs/root/opt/raven_libs/environments/raven_libs_profile
cat - > $PROFILE_FILE << RAVEN_PROFILE
source /opt/raven_libs/bin/activate
RAVEN_PROFILE

chmod +x $PROFILE_FILE
mkdir -p $HOME/raven_libs/scripts
cat - > $HOME/raven_libs/scripts/preflight <<PREFLIGHT
#!/bin/bash
rm -Rf /opt/raven_libs/
PREFLIGHT
chmod +x $HOME/raven_libs/scripts/preflight


rm -Rf raven_pip.pkg
pkgbuild --root $HOME/raven_libs/root --identifier raven_libs  --scripts $HOME/raven_libs/scripts raven_pip.pkg

