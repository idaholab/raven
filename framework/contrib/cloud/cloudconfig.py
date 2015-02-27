"""
PiCloud configuration settings
Two settings are stored in this file - the baseLocation for cloud AND the config file location
All other variables are stored in a config file managed by this

Copyright (c) 2009 `PiCloud, Inc. <http://www.picloud.com>`_.  All rights reserved.

email: contact@picloud.com

The cloud package is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This package is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this package; if not, see 
http://www.gnu.org/licenses/lgpl-2.1.html
"""


from __future__ import with_statement
import os
import distutils
import distutils.dir_util

#Location for all cloud configuration
if os.name == "nt":
    baselocation = os.environ.get('APPDATA')
    if baselocation:
        baselocation = os.path.join(baselocation, 'picloud')
    else:
        baselocation =  os.path.join('~', '.picloud')
else:
    baselocation = os.path.join('~', '.picloud')


#name of configuration file:
configname = 'cloudconf.py'

_needsWrite = False

genHidden = False #if true config will generate hidden variables

from .util import configmanager

def get_config_value(section, varname, default, comment=None, hidden=False):
    
    """
    Return value associated with a varname found in section
    If varname is not found, return defaultval and add varname
    If varname found, cast value to type of default
    """       
    typ = type(default)
    try:
        entry = config.get(section, varname, comment)
        return typ(entry)
        
    except (configmanager.NoOptionError), e:        
        if hidden and not genHidden:
            config.hiddenset(section,varname,default,comment)            
            return default
        config.set(section,varname,default,comment)        
        return default
    except ValueError, e:
        import logging
        log = logging.getLogger('Cloud') #might not work if logging not yet initialized
        log.warning('Option %s.%s in cloudconf.py must have type %s. Reverting to default' % (section, varname, typ.__name__))
        return default    

"""Sections defined below"""    
def account_configurable(varname,default,comment=None, hidden=False):
    return get_config_value('Account',varname,default,comment,hidden)
    
def logging_configurable(varname,default,comment=None, hidden=False):
    return get_config_value('Logging',varname,default,comment,hidden)

def mp_configurable(varname,default,comment=None, hidden=False):
    return get_config_value('Multiprocessing',varname,default,comment,hidden)

def simulation_configurable(varname,default,comment=None, hidden=False):
    return get_config_value('Simulation',varname,default,comment,hidden)

def transport_configurable(varname, default, comment=None, hidden=False):
    return get_config_value('Transport',varname,default,comment,hidden)
    


def flush_config():
    """
    Write settings to file
    """
    conf_path = os.path.join(fullconfigpath,configname)
    try:
        with open(conf_path, 'w') as configfile:    
            config.write(configfile)
    except IOError, e:
        import logging
        log = logging.getLogger('Cloud') #might not work if logging not yet initialized
        log.exception('Could not write %s.', conf_path)

    

""" Setup
"""
import os
import distutils.errors

#test

config = configmanager.ConfigManager()
fullconfigpath = os.path.expanduser(baselocation)
try:
    newdir = distutils.dir_util.mkpath(fullconfigpath) #ensure path exists
except distutils.errors.DistutilsFileError:
    #this will fail
    config.read(os.path.join(fullconfigpath,configname))

else:
    if newdir:
        os.chmod(newdir[-1], 0700)
        
    if not config.read(os.path.join(fullconfigpath,configname)):
        _needsWrite = True
    #cloudLog.debug("Cloud Configuration imported")
