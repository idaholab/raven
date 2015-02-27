"""
Interface for cloud - the clientside module for PiCloud

Sample usage::

    import cloud
    jid = cloud.call(lambda: 3*3)
    >> Returns a job identifier
    cloud.result(jid)
    >> Returns 9

This will run the function lambda: 3*3 on PiCloud's
cluster, and return the result. Most functions, even
user-defined ones, can be passed through cloud.call

For cloud to work, you must first run 'picloud setup' in your shell.

Alternatively, you can use the simulator by setting use_simulator to True 
in cloudconf.py or running cloud.start_simulator()
"""
"""
Copyright (c) 2010 `PiCloud, Inc. <http://www.picloud.com>`_.  All rights reserved.

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

import sys
import imp
#import threading

#versioning:
from .versioninfo import release_version
__version__ = release_version

import logging

#Input hooks
class ImportHook(object):
    def __init__(self):
        self.mods = []  #Reloading mods in order will reload entire cloud
    
    def load_module(self, fullname):
        """This is called by the system to load a module
            We use it to track mopdules after their code is loaded"""        
        parentmod, ext, submod = fullname.rpartition('.')
        if parentmod:
            path = sys.modules[parentmod].__path__
        else:
            path = ""
    
        file, filename, tuple = imp.find_module(submod, path)
        
        #This may load additional modules which are appended before this module in self.mods
        imp.acquire_lock()
        try:
            mod = imp.load_module(fullname, file, filename, tuple)
        except ImportError, i:
            raise
        finally:
            imp.release_lock()  
        
        self.mods.append(fullname)        
        return mod
    
    
    def find_module(self, fullname, path = None):
        """This is called by the system to find a module.
        cloud.* modules are hooked into our system"""
        #print 'find %s -- %s' % (fullname, threading.current_thread().ident)
        
        if False:
            v = logging.getLogger('Cloud')
            try:
                v.debug('IMPORT: Attempting to load %s from %s', fullname, path)
            except Exception: # race condition with multiprocessing
                pass
        
        if fullname.startswith('cloud') and 'server' not in fullname:
            self.path = path
            parentmod, ext, submod = fullname.rpartition('.')
            if parentmod:
                path = sys.modules[parentmod].__path__
            else:
                path = ""
            try: #Fails due to attempting relative import of python packages, e.g cloud.os
                grp = imp.find_module(submod, path)
                if not grp: #import error
                    return None
            except ImportError: 
                return None 
            else:
                return self
        else:
            return None
    
_modHook = ImportHook()    
sys.meta_path.append(_modHook) #registers our import hook    

from . import cloudconfig as cc
import cloudinterface
from . import cloud

__cloud = None
__type = None
__immutable = False

#function bindings
call = None
map = None
status = None
join = None
result = None
kill = None
is_simulated = None
running_on_cloud = None
connection_info = None

__all__ = ["call", "map", "status", "result", "iresult", "is_simulated", "join", "kill",
            "delete", "info", "connection_info", "running_on_cloud", "setkey", "finished_statuses",
             "start_simulator", "config", "getconfigpath", "close","c1","c2","m1"]

c = """The Cloud Simulator simulates cloud.* functions locally, allowing for quicker debugging.
If this is enabled, all cloud.* functions will be run locally, rather than on PiCloud. See web documentation."""
__useSimulator =  cc.account_configurable('use_simulator',
                                          default=False,
                                          comment=c)

def _launch_cloud():
    cloudinterface._setcloud(sys.modules[__name__], 'simulated' if __useSimulator else 'network', restart=True)
    
_launch_cloud()

from .cloud import CloudException, CloudTimeoutError

def start_simulator(force_restart = False):
    """
    Simulate cloud functionality locally.
    If *force_restart* or not already in simulation mode, restart cloud and enter simulation.
    In simulation mode, the cloud will be run locally, on multiple processors, via 
    the multiprocessing library.
    Additional logging information will be enabled.
    For more information, see 
    `PiCloud documentation <http://docs.picloud.com/cloud_simulator.html>`_
    """

    cloudinterface._setcloud(sys.modules[__name__], 'simulated', restart=force_restart)

    
def setkey(api_key, api_secretkey=None, server_url=None, restart=False, immutable=False):
    """
    Connect cloud to your PiCloud account, given your *api_key*.
    Your *api_key* is provided by PiCloud on the 
    `API Keys <http://www.picloud.com/accounts/apikeys/>`_ section of the PiCloud website.
    
    The *api_secretkey* is generally stored on your machine.  However, if you have not previously used 
    this api_key or selected it in 'picloud setup', you will need to provide it.
    
    *server_url* specifies the PiCloud server to connect to.  Leave this blank to auto-resolve servers.

    *restart* forces the cloud to reconnect
    
    This command will disable the simulator if it is running.
    """
    
    cloudinterface._setcloud(sys.modules[__name__], 'network', api_key, api_secretkey, server_url, restart, immutable)

def set_dependency_whitelist(whitelist):
    """            
    By default all relevant dependencies found will be transported to PiCloud
    In secure scenarios, you may wish to restrict the dependencies that may be transferred.
    
    whitelist should be a list consisting of module names that can be transported.
    
    Example::
    
        ['mypackage','package2.mymodule'] 
        
        * Allows mypackage, package2.mymodule
        * Disallows foo_module, package2.mymodule2 

    Set to None to re-enable full automatic dependency magic
    
    .. warning::
        
        Dependency whitelist will reset if setkey() or start_simulator() is run
    """ 
    from .transport.adapter import DependencyAdapter
    from .transport.network import HttpConnection
    
    # must have a connected network connection to create a dependency manager
    netcon = _getcloudnetconnection()
    adapter = netcon.adapter
    
    adapter._create_dependency_manager(whitelist)


def _getcloud():
    """Return internal cloud object. Only for internal use"""
    return __cloud

def _getcloudnetconnection():
    """Returns cloud connection if it is an HttpConnection. else raise Exception"""
    from .transport.adapter import SerializingAdapter
    from .transport.network import HttpConnection
    if not isinstance(__cloud.adapter, SerializingAdapter):
        raise Exception('Unexpected cloud adapter being used')
    
    conn = __cloud.adapter.connection
    if isinstance(conn, HttpConnection):
        __cloud._checkOpen()
        return conn
    else:
        raise RuntimeError('Cannot use this functionality in simulation')

def getconfigpath():
    """
    Return the directory where PiCloud configuration and logging files are stored.    
    Configuration/logs are stored on a per-user basis
    """
    return _cc.fullconfigpath

"""Read/Write configuration file"""
from .util.configmanager import ConfigSettings
config = ConfigSettings(cc.config)  #sets cc.config values
if cc._needsWrite:
    cc.flush_config()
    cc._needsWrite = False    

del ConfigSettings
_cc = cc
del cc

try:
    from . import mp as mp
except ImportError:
    try:
        import multiprocessing
    except ImportError: #if mp error, fall back
        cloud.cloudLog.warn('Multiprocessing is not installed. Cloud.mp will be disabled')
    else:
        raise #something else has gone wrong
    
from . import account   #cloud.account
from . import cron      #cloud.cron

"""
if __cloud.running_on_cloud():
    # use server-side implementation of cloud.bucket
    from . import _server_bucket as bucket   #cloud.bucket
else:   
    from . import bucket    #default loud.bucket
"""    

from . import shell     #cloud.shell
from . import queue     #cloud.queue

from . import files     #cloud.files (DEPRECATED)