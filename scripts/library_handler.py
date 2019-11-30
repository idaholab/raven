# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Library management for RAVEN installation and operation.
Adopted from RavenUtils.py
"""
import os
import sys
import platform
import argparse
import subprocess
from collections import OrderedDict

from update_install_data import loadRC
import plugin_handler as pluginHandler

# python changed the import error in 3.6
if sys.version_info[0] == 3 and sys.version_info[1] >= 6:
  impErr = ModuleNotFoundError
else:
  impErr = ImportError

# python 2.X uses a different capitalization for configparser
try:
  import configparser
except impErr:
  import ConfigParser as configparser

try:
  # python 3.8+ includes this in std lib
  import importlib_metadata
  usePackageMeta = True
except impErr:
  # the old way to check libs and versions uses subprocess instead
  usePackageMeta = False

# some libs are called differently in conda and within python interpreter
## only needed for the subprocess way to check libs!
## mapping is conda name: python import name
libAlias = {'scikit-learn': 'sklearn',
            'netcdf4': 'netCDF4',
            'pyside2': 'PySide2',
            'lazy-import': 'lazy_import',
           }

# some bad actors can't use the metadata correctly
# and so need special treatment
# -> see findLibAndVersion
metaExceptions = ['pyside2', 'AMSC']

skipChecks = ['python', 'hdf5', 'swig', 'nomkl']

# load up the ravenrc if it's present
## TODO we only want to do this once, but does it need to get updated?
rcFile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.ravenrc'))
if os.path.isfile(rcFile):
  rcOptions = loadRC(rcFile)
else:
  rcOptions = {}

#############
#    API    #
#############
def inPython3():
  """
    Legacy method for consistency. Returns True, since we're python 3 now.
    Leaving this here for Python 4 or similar futures.
    @ In, None
    @ Out, inPython3, True
  """
  return True

def checkVersions():
  """
    Returns true if versions should be checked.
    @ In, None
    @ Out, checkVersions, bool, if true versions should be checked.
  """
  return os.environ.get("RAVEN_IGNORE_VERSIONS", "0") != "1"

def checkLibraries(buildReport=False):
  """
    Looks for required libraries, and their matching QA versions.
    @ In, buildReport, bool, optional, if True then report all libraries instead of just problems
    @ Out, missing, list(tuple(str, str)), list of missing libraries and needed versions
    @ Out, notQA, list(tuple(str, str, str)), mismatched versions as (libs, need version, found version)
  """
  missing = []
  notQA = []
  plugins = pluginHandler.getInstalledPlugins()
  need = getRequiredLibs(plugins=plugins)
  messages = []
  for lib, needVersion in need.items():
    # some libs aren't checked from within python
    if lib in skipChecks:
      continue
    found, msg, foundVersion = checkSingleLibrary(lib, version=needVersion)
    if not found:
      missing.append((lib, needVersion))
      continue
    if needVersion is not None and foundVersion != needVersion:
      notQA.append((lib, needVersion, foundVersion))
    if buildReport:
      messages.append((lib, found, msg, foundVersion))
  if buildReport:
    return messages
  return missing, notQA

def checkSingleLibrary(lib, version=None, useImportCheck=False):
  """
    Looks for library and check if available.
    @ In, lib, str, name of library
    @ In, version, str, optional, version to check (e.g. '0.12.2')
    @ In, useImportCheck, bool, optional, force to use manual check
    @ Out, found, bool, True if library accessible
    @ Out, msg, str, message regarding attempted import
    @ Out, foundVersion, version detected (or None if not required or found)
  """
  # use the faster package checking if possible
  ## this avoids actually importing the modules
  if usePackageMeta and not useImportCheck:
    found, msg, foundVersion = findLibAndVersion(lib, version=version)
  # otherwise, use the slower subprocess method
  else:
    found, msg, foundVersion = findLibAndVersionSubprocess(lib, version=version)
  return found, msg, foundVersion

def findLibAndVersion(lib, version=None):
  """
    Attempts to load a library and check its version if given.
    @ In, lib, str, conda name of lib (may differ from python's name)
    @ In, version, str, optional, version to check (e.g. '0.12.2') (unused, kept for consistency with subprocess version)
    @ Out, found, bool, True if successful import
    @ Out, output, str, message regarding attempted import
    @ Out, foundVersion, version detected (or None if not required or found)
  """
  # well-behaved lib version checks
  if lib not in metaExceptions:
    try:
      foundVersion = importlib_metadata.version(lib)
      found = True
      output = 'Library found.'
    except importlib_metadata.PackageNotFoundError:
      found = False
      foundVersion = None
      output = 'Library not found.'
  # bad actors
  ## FIXME: if updating pyside2, check if it can be found with importlib_metadata!
  ## Was not possible in version 5.13.1 - talbpaul
  else:
    if lib == 'pyside2':
      try:
        import PySide2._config as ps2c
        found = True
        foundVersion = ps2c.version
        output = 'Library found.'
      except impErr:
        found = False
        foundVersion = None
        output = 'Library not found.'
    elif lib == 'AMSC':
      # FIXME improve AMSC setup.py so it's compatible with importlib_metadata!
      return findLibAndVersionSubprocess('AMSC')
    else:
      raise NotImplementedError('Library "{}" on exception list, but no exception implemented!'.format(lib))
  return found, output, foundVersion

def findLibAndVersionSubprocess(lib, version=None):
  """
    Attempts to load a library and check its version if given.
    Note this is a slower method, only use if importlib_metadata isn't available!
    -> this version actually loads up each module, which is slow.
    @ In, lib, str, conda name of lib (may differ from python's name)
    @ In, version, str, optional, version to check (e.g. '0.12.2')
    @ Out, found, bool, True if successful import
    @ Out, output, str, message regarding attempted import
    @ Out, foundVersion, version detected (or None if not required or found)
  """
  ## NOTE:
  # suprocesses help keep the import checks clean, and assure we use the version
  # of python that we want to. Strictly speaking, they should not be necessary
  # in the future.
  ## end note
  # fix python name if on nt (windows conda, python.org install)
  python = 'python' if os.name == 'nt' else 'python3'
  # check for loadability first
  command = 'import {mod}; print({mod})'.format(mod=libAlias.get(lib, lib))
  try:
    foundExists = subprocess.check_output([python, '-c', command], universal_newlines=True)
  except subprocess.CalledProcessError:
    return False, 'Failed to find module "{}" (known in python as "{}")'.format(lib, libAlias.get(lib, lib)), None
  # if successfully loaded, check version if needed
  if version is not None:
    command = 'import {mod}; print({mod}.__version__)'.format(mod=libAlias.get(lib, lib))
    try:
      foundVersion = subprocess.check_output([python, '-c', command], universal_newlines=True).strip()
    except subprocess.CalledProcessError:
      # if module doesn't have a __version__, report as not available
      foundVersion = None
  # if no version required, no need to report the version
  else:
      foundVersion = None
  return True, foundExists, foundVersion

def getRequiredLibs(useOS=None, installMethod=None, addOptional=False, limit=None, plugins=None):
  """
    Assembles dictionary of required libraries.
    @ In, useOS, str, optional, if provided then assume given operating system
    @ In, installMethod, str, optional, if provided then assume given install method
    @ In, addOptional, bool, optional, if True then add optional libraries to list
    @ In, limit, list(str), optional, limit sections that are read in
    @ In, plugins, list(tuple(str,str)), optional, plugins (name, location) that should be added to
                   the required libs
    @ Out, libs, dict, dictionary of libraries {name: version}
  """
  mainConfigFile = os.path.abspath(os.path.expanduser(os.path.join(os.path.dirname(__file__),
                                                                   '..', 'dependencies.ini')))
  config = _readDependencies(mainConfigFile)
  opSys = _getOperatingSystem(override=useOS)
  install = _getInstallMethod(override=installMethod)
  libs = _parseLibs(config, opSys, install, addOptional=addOptional, limit=limit)
  # extend config with plugin libs
  for pluginName, pluginLoc in plugins:
    pluginConfigFile = os.path.join(pluginLoc, 'dependencies.ini')
    if os.path.isfile(pluginConfigFile):
      pluginConfig = _readDependencies(pluginConfigFile)
      pluginLibs = _parseLibs(pluginConfig, opSys, install, addOptional=addOptional, limit=limit)
      pluginLibs = _checkForUpdates(libs, pluginLibs, pluginName)
      libs.update(pluginLibs)
  return libs

def _checkForUpdates(libs, pluginLibs, pluginName):
  """
    Checks requested lib updates for conflicts
    @ In, libs, dict, existing libs: versions
    @ In, pluginLibs, dict, requested changes as libs: versions
    @ In, pluginName, str, name of plugin
    @ Out, pluginLibs, dict, modified plugin library
  """
  # make sure no changing of required versions is done
  toRemove = []
  for pluginLibName, pluginVersion in pluginLibs.items():
    if pluginLibName in libs: # NOTE: do not use libs.get(~,None) here, since many libs map to None
      pinned = libs[pluginLibName]
      # a couple things could be happening here:
      #  -> no change requested
      if pinned == pluginVersion:
          # lib already exists in list; no change needed
          toRemove.append(pluginLibName)
      #  -> the existing unpinned version is being pinned (okay, update)
      elif pinned is None:
        # NOTE: assumed: pluginVersion is also not None, else it would be caught in 'if =='
        pass # leave in the dict to be updated
      #  -> the existing lib (pinned or not) is requested without a version (okay, no action)
      elif pluginVersion is None:
        # pinned version supercedes requirement, so no change needed
        toRemove.append(pluginLibName)
      #  -> the existing lib (pinned) is being given a new version (not okay, error)
      else:
        # we already know:
        #   pinned is not None
        #   pluginVersion is not None
        #   pluginVersion != pinned
        # therefore we have a conflict
        raise ValueError(('Plugin "{plug}" is trying to change already-pinned version of library '+
                          '"{lib}" from "{pin}" to "{plugVer}"').format(plug=pluginName,
                                                                        lib=pluginLibName,
                                                                        pin=libs[pluginLibName],
                                                                        plugVer=pluginVersion))
  # clear libs that don't need updating
  for name in toRemove:
    del pluginLibs[name]
  return pluginLibs

#############
#   UTILS   #
#############
def _getOperatingSystem(override=None):
  """
    Determine the operating system in use.
    @ In, override, str, optional, use given OS if valid
    @ Out, os, str, name of operating system class
  """
  valid = ['windows', 'mac', 'linux']
  if override is not None:
    if override.lower() not in valid:
      raise TypeError('Library Handler: Provided override OS not recognized: "{}"! Acceptable options: {}'.format(override, valid))
    return override.lower()
  # since no suggestion given, try to determine
  osName = platform.system().lower()
  if osName in ['linux', 'linux-gnu']:
    return 'linux'
  elif osName in ['windows', 'msys', 'cygwin']:
    return 'windows'
  elif osName in ['darwin']:
    return 'mac'
  else:
    # TODO should we just default to linux here? Leaving as error to check test machines for now.
    raise TypeError('Unrecognized platform system: "{}"'.format(osName))

def _getInstallMethod(override=None):
  """
    Determine the install method (conda or pip) (custom?)
    @ In, override, str, optional, use given method if valid
    @ Out, install, str, type of install
  """
  valid = ['conda', 'pip'] #custom?
  if override is not None:
    if override.lower() not in valid:
      raise TypeError('Library Handler: Provided override install method not recognized: "{}"! Acceptable options: {}'.format(override, valid))
    return override.lower()
  # check ravenrc for how libs were installed
  if rcOptions:
    return rcOptions.get('INSTALLATION_MANAGER', 'conda').lower()
  # no suggestion given, so we assume conda
  return 'conda'

def _parseLibs(config, opSys, install, addOptional=False, limit=None, plugins=None):
  """
    Parses config file to get libraries to install, using given options.
    @ In, config, configparser.ConfigParser, read-in dependencies
    @ In, opSys, str, operating system (not checked)
    @ In, install, str, installation method (not checked)
    @ In, addOptional, bool, optional, if True then include optional libraries
    @ In, limit, list(str), optional, if provided then only read the given sections
    @ In, plugins, list(tuple(str,configParser.configParser)), optional, plugins (name, config)
                   that should be added to the parsing
    @ Out, libs, dict, dictionary of libraries {name: version}
  """
  libs = OrderedDict()
  # get the main libraries, depending on request
  for src in ['core', 'forge', 'pip']:
    if config.has_section(src) and (True if limit is None else (src in limit)):
      _addLibsFromSection(config.items(src), libs)
  # os-specific are part of 'core' right now
  if config.has_section(opSys) and (True if limit is None else ('core' in limit)):
    _addLibsFromSection(config.items(opSys), libs)
  # optional are part of 'core' right now, but leave that up to the requester?
  if addOptional and config.has_section('optional'):
    _addLibsFromSection(config.items('optional'), libs)
  if install == 'pip' and config.has_section('pip-install'):
    _addLibsFromSection(config.items('pip-install'), libs)
  return libs

def _addLibsFromSection(configSection, libs):
  """
    Reads in libraries for a section of the config.
    @ In, configSection, dict, libs: versions
    @ In, libs, dict, libraries tracking dict
    @ Out, None (changes libs in place)
  """
  for lib, version in configSection:
    #if lib not in configSection:
    #  return
    if version == 'remove':
      libs.pop(lib, None)
    else:
      # python 3 fix to work with python 2 syntax
      if version is not None and version.strip() == '':
        version = None
      libs[lib] = version

def _readDependencies(initFile):
  """
    Reads in the library list using config parsing.
    @ In, None
    @ Out, configparser.ConfigParser, configurations read in
  """
  config = configparser.ConfigParser(allow_no_value=True)
  config.read(initFile)
  return config

if __name__ == '__main__':
  mainParser = argparse.ArgumentParser(description='RAVEN Library Handler')
  mainParser.add_argument('--os', dest='useOS',
        choices=('windows', 'mac', 'linux'), default=None,
        help='Determines the operating system for which the library configuration will be built.')
  mainParser.add_argument('--plugins', dest='usePlugins', action='append',
        help='Lists the plugins (or none or all) whose libraries should be included. Default: all.')
  mainParser.add_argument('--optional', dest='addOptional', action='store_true',
        help='Include optional libraries in configuration.')
  subParsers = mainParser.add_subparsers(help='Choose library installer.', dest='installer')

  condaParser = subParsers.add_parser('conda', help='use conda as installer')
  condaParser.add_argument('--action', dest='action',
        choices=('create', 'install', 'list'), default='install',
        help='Chooses whether to (create) a new environment, (install) in existing environment, ' +
             'or (list) installation libraries.')
  condaParser.add_argument('--subset', dest='subset',
        choices=('core', 'forge', 'pip'), default='core',
        help='Use subset of installation libraries, divided by source.')

  pipParser = subParsers.add_parser('pip', help='use pip as installer')
  pipParser.add_argument('--action', dest='action', choices=('install', 'list'), default='install',
        help='Chooses whether to (install) in current environment, or ' +
             '(list) installation libraries.')

  manualParser = subParsers.add_parser('manual', help='provide LaTeX manual list')

  args = mainParser.parse_args()

  # load library environment name
  envName = os.getenv('RAVEN_LIBS_NAME', 'raven_libraries')

  if args.useOS is None:
    args.useOS = _getOperatingSystem()
  if args.usePlugins is None:
    args.usePlugins = ['all']

  plugins = [] # list of [(name, location), (name, location)] for each plugin
  if 'all' in args.usePlugins:
    plugins = pluginHandler.getInstalledPlugins()
  elif 'none' in args.usePlugins:
    pass # nothing to do
  else:
    # separate out comma-separated
    usePlugins = []
    for pluginArg in args.usePlugins:
      usePlugins.extend(pluginArg.split(','))
    # store plugin names and locations
    missing = []
    for pluginName in usePlugins:
      if pluginName.strip() == '':
        continue
      loc = pluginHandler.getPluginLocation(pluginName)
      if loc is None:
        print('ERROR (library_handler): Plugin "{}" has not been installed!'.format(pluginName))
        missing.append(pluginName)
      else:
        plugins.append((pluginName, loc))
    if missing:
      raise IOError('During library installation, the following requested plugin libraries '+
                    ' were not found: {}'.format(', '.join(missing)))

  ### Il Grande Albero Decisionale
  # "optional" and "os" are passed through
  if args.installer == 'manual':
    # compile the LaTeX lib list
    libs = getRequiredLibs(useOS=args.useOS, installMethod='conda',
                           addOptional=args.addOptional, plugins=plugins)
    msg = '\\begin{itemize}\n'
    for lib, version in libs.items():
      msg += '  \\item {}{}\n'.format(
             lib.replace('_', '\\_'), ('' if version is None else '-'+version))
    msg += '\\end{itemize}'
    print(msg)
  else:
    # provide an installation command
    preamble = '{installer} {action} {args} '
    if args.installer == 'conda':
      installer = 'conda'
      equals = '='
      actionArgs = '--name {env} -y {src}'
      # which part of the install are we doing?
      if args.subset == 'core':
        # no special source
        src = ''
        addOptional = args.addOptional
        limit = ['core']
      elif args.subset == 'forge':
        # take libs from conda-forge
        src = '-c conda-forge '
        addOptional = False
        limit = ['forge']
      elif args.subset == 'pip':
        src = ''
        installer = 'pip'
        actionArgs = ''
        addOptional = False
        limit = ['pip']
      libs = getRequiredLibs(useOS=args.useOS,
                             installMethod='conda',
                             addOptional=addOptional,
                             limit=limit,
                             plugins=plugins)
      # conda can create, install, or list
      if args.action == 'create':
        action = 'create'
      elif args.action == 'install':
        action = 'install'
      elif args.action == 'list':
        preamble = ''
      actionArgs = actionArgs.format(env=envName, src=src)
    elif args.installer == 'pip':
      installer = 'pip3'
      equals = '=='
      actionArgs = ''
      libs = getRequiredLibs(useOS=args.useOS,
                             installMethod='pip',
                             addOptional=args.addOptional,
                             limit=None,
                             plugins=plugins)
      if args.action == 'install':
        action = 'install'
      elif args.action == 'list':
        preamble = ''

    preamble = preamble.format(installer=installer, action=action, args=actionArgs)
    libTexts = ' '.join(['{lib}{ver}'
                         .format(lib=lib,
                                 ver=('{}{}'.format(equals, ver) if ver is not None else ''))
                         for lib, ver in libs.items()])
    print(preamble + libTexts)
