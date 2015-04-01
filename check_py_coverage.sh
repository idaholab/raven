#!/bin/bash
BUILD_DIR=${BUILD_DIR:=$HOME/raven_libs/build}
INSTALL_DIR=${INSTALL_DIR:=$HOME/raven_libs/pylibs}
PYTHON_CMD=${PYTHON_CMD:=python}
JOBS=${JOBS:=1}
mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
DOWNLOADER='curl -C - -L -O '
SCRIPT_DIRNAME=`dirname $0`
SCRIPT_DIR=`(cd $SCRIPT_DIRNAME; pwd)`

ORIGPYTHONPATH="$PYTHONPATH"

update_python_path ()
{
    if ls -d $INSTALL_DIR/*/python*/site-packages/
    then
        export PYTHONPATH=`ls -d $INSTALL_DIR/*/python*/site-packages/`:"$ORIGPYTHONPATH"
    fi
}

update_python_path
PATH=$INSTALL_DIR/bin:$PATH


if which coverage
then
    echo coverage already available, skipping building it.
else
    if curl http://www.energy.gov > /dev/null
    then
       echo Successfully got data from the internet
    else
       echo Could not connect to internet
    fi

    cd $BUILD_DIR
    $DOWNLOADER https://pypi.python.org/packages/source/c/coverage/coverage-3.7.1.tar.gz #md5=67d4e393f4c6a5ffc18605409d2aa1ac
    tar -xvzf coverage-3.7.1.tar.gz
    cd coverage-3.7.1
    (unset CC CXX; $PYTHON_CMD setup.py install --prefix=$INSTALL_DIR)
fi

update_python_path

cd $SCRIPT_DIR

EXTRA='--rcfile=.coveragerc --source=../../framework -a --omit=../../framework/contrib/*'
cd tests/framework
#coverage help run


coverage erase
#skip test_rom_trainer.xml
for I in test_simple.xml test_output.xml test_branch.xml test_preconditioned_det.xml test_push_into_hdf5.xml test_rom_trainer_no_normalization.xml test_rom_train_from_already_dumped_HDF5.xml test_FullFactorial_Sampler.xml test_ResponseSurfaceDesign_Sampler.xml test_Grid_Sampler.xml test_random.xml test_LHS_Sampler.xml test_Grid_Sampler_Bison.xml test_LHS_Sampler_Bison.xml test_LHS_Sampler_Raven.xml test_Grid_Sampler_Raven.xml test_Lorentz.xml test_BasicStatistics.xml test_LimitSurface.xml test_CreateInternalObjFromCSVs.xml test_bison_mc_simple.xml test_custom_mode.xml test_iostep_load.xml test_safest_point.xml test_safest_point_cdf.xml test_externalPostProcessor.xml test_adaptive_det_simple.xml test_external_reseed.xml test_stoch_poly.xml test_indexsets.xml test_stochpoly_interp.xml test_io_ROM_pickle.xml test_relap5_code_interface.xml test_cc_stats.xml test_generic_interface.xml test_sobol_sampler.xml
do
    echo Running $I
    coverage run $EXTRA ../../framework/Driver.py  $I
done
coverage run $EXTRA ../../framework/TestDistributions.py
coverage run $EXTRA ../../framework/Driver.py test_relap5_code_interface.xml interfacecheck
coverage html

