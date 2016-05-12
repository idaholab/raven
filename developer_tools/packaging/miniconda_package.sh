export https_proxy=$http_proxy
export INSTALL_DIR=/opt/raven_libs

rm -Rvf $INSTALL_DIR
rm Miniconda-latest-MacOSX-x86_64.sh
curl -C - -L -O https://repo.continuum.io/miniconda/Miniconda-latest-MacOSX-x86_64.sh
chmod +x Miniconda-latest-MacOSX-x86_64.sh
./Miniconda-latest-MacOSX-x86_64.sh -b -p $INSTALL_DIR
export PATH=$INSTALL_DIR/bin:$PATH
#Call RavenUtils to return the conda create command with the qa'd versions
#`python ../../scripts/TestHarness/testers/RavenUtils.py --conda-create`
#conda install -y numpy hdf5 h5py scipy scikit-learn matplotlib swig
conda install -y numpy=1.9.2 hdf5=1.8.14 h5py=2.5.0 scipy=0.15.1 scikit-learn=0.16.1 matplotlib=1.4.3 swig=3.0.2
conda remove -y qt

rm -Rvf $HOME/raven_libs/root
mkdir -p $HOME/raven_libs/root/opt
mv $INSTALL_DIR $HOME/raven_libs/root/opt
mkdir -p $HOME/raven_libs/root/opt/raven_libs/environments
PROFILE_FILE=$HOME/raven_libs/root/opt/raven_libs/environments/raven_libs_profile
cat - > $PROFILE_FILE << RAVEN_PROFILE
export PATH="/opt/raven_libs/bin:\$PATH"
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

rm -Rf raven_miniconda.pkg
pkgbuild --root $HOME/raven_libs/root --identifier raven_libs  --scripts $HOME/raven_libs/scripts raven_miniconda.pkg

#Create dmg file.
rm -f raven_libs_base.dmg raven_miniconda.dmg
hdiutil create -size 850m -fs HFS+ -volname "Raven Libraries" raven_libs_base.dmg
hdiutil attach raven_libs_base.dmg
cp -a raven_miniconda.pkg /Volumes/Raven\ Libraries
hdiutil detach /Volumes/Raven\ Libraries/
hdiutil convert raven_libs_base.dmg -format UDZO -o raven_miniconda.dmg
