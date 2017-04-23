"""
Defines all Cloud Adapters - objects that control lower-level behavior of the cloud
There is a one to one instance mapping between a cloud and its adapter

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

import sys
import time
import types
import threading
from functools import partial
from itertools import izip, imap, count

from ..cloud import CloudException
from .. import serialization
from .. import cloudconfig as cc
from ..cloudlog import cloudLog
from ..util import OrderedDict


class Adapter(object):
    """
    Abstract class to deal with lower-level cloud operations
    """
          
    _isopen = False
    
    @property
    def opened(self): 
        """Returns whether the adapter is open"""
        return self._isopen
    
    def open(self):
        """called when this adapter is to be used"""
        if self.opened:
            raise CloudException("%s: Cannot open already-opened Adapter", str(self))
        if not self._cloud.opened:
            self._cloud.open()
    
    def close(self):
        """called when this adapter is no longer needed"""
        if not self.opened:
            raise CloudException("%s: Cannot close a closed adapter", str(self))
        self._isopen = False
        
    @property
    def cloud(self):
        return self._cloud
    
    def needs_restart(self, **kwargs):
        """Called to determine if the cloud must be restarted due to different adapter parameters"""
        return False
        
    def call(self, params, func, *args):
        raise NotImplementedError
    
    def jobs_join(self, jids, timeout = None):
        """
        Allows connection to manage joining
        If connection manages joining, it should return a list of statuses  
        describing the finished job
        Else, return False
        """
        return False
    
    def jobs_map(self, params, func, argList):
        """Treat each element of argList as func(*element)"""
        raise NotImplementedError
    
    def jobs_result(self, jids):
        raise NotImplementedError

    def jobs_kill(self, jids):
        raise NotImplementedError
    
    def jobs_delete(self, jids):
        raise NotImplementedError    
    
    def jobs_info(self, jids, info_requested):
        raise NotImplementedError
    
    def is_simulated(self):        
        raise NotImplementedError
    
    def connection_info(self):
        return self.connection.connection_info()
    
    def dep_snapshot(self):
        """This function, called downstream from the network connection object, 
        deals with calling the dependency checking functions
        Note that it can be called in between successive maps"""
        pass #do nothing by default

        

"""
TheSerializingAdapter is responsible for serializing data passed into args
It then passes data onto its CloudConnection handler
It can (optionally) log serializations
"""

class SerializingAdapter(Adapter):
    
    #config for pickle debugging        
    c = """Should serialization routines track debugging information?
    If this is on, picklingexceptions will provide a 'pickle trace' that identifies what member of an object cannot be pickled"""
    serializeDebugging =  \
        cc.logging_configurable('serialize_debugging',
                                          default=True,
                                          comment=c, hidden=True)
    c = """Should all object serialization meta-data be logged to the serialize_logging_path?""" 
    serializeLogging = \
        cc.logging_configurable('serialize_logging',
                                          default=False,
                                          comment=c)      
    if serializeLogging and not serializeDebugging:
        serializeDebugging = True
        cloudLog.warning("serialize_logging implies serialize_debugging. Setting serialize_debugging to true.")
    
    
    c = """Maximum amount of data (in bytes) that can be sent to the PiCloud server per function or function argument. 
    Value may be raised up to 16 MB"""
    max_transmit_data = \
        cc.transport_configurable('max_transmit_data',
                                  default=1000000,
                                  comment=c)
    max_transmit_data = min(max_transmit_data,16000000)
        
    
    
    #live:
    _report = None  
    _isSlave = False #True if this adapter is a slave (it's connection is in a master process)
                  
        
    def __init__(self, connection, isSlave = False):
        self.connection = connection 
        connection._adapter = self
        self.opening = False
        self.openLock = threading.RLock()            
        self._isSlave = isSlave
        
    def init_report(self, subdir = ""):
        try:
            self._report = serialization.SerializationReport(subdir)
        except IOError, e:
            cloudLog.warn('Cannot construct serialization report directory. Error is %s' % str(e))
            self._report = None
            self.serializeLogging = False #disable logging
        
    @property
    def report(self):
        return self._report
    
    @property
    def isSlave(self):
        return self._isSlave

    def _configure_logging(self):
        #Seperated from open to allow for dynamic changes at runtime
        
        #set up cloud serializer object
        self.min_size_to_save =  0 if self.serializeLogging else (self.max_transmit_data / 3)   
        #reporting:                
        if not self._report:
            if self.serializeLogging or self.connection.force_adapter_report():
                if self._isSlave:
                    self.init_report()
                    self._report.logPath = self.connection.get_report_dir() 
                else:
                    self.init_report(self.connection.report_name())            
            else:
                self._report = None  

    def getserializer(self, serialization_level = 0):
        """Return the correct serializer based on user settings and serialization_level"""
        if serialization_level >= 2:
            return serialization.Serializer
        elif self.serializeDebugging and serialization_level == 0:
            return serialization.DebugSerializer
        else:
            return serialization.CloudSerializer        
        
    
    def open(self):      
        with self.openLock:
            if self._isopen or self.opening:
                return

            self.opening = True
            Adapter.open(self)  
                                                
            try:                 
                #always try to open               
                if self._isSlave:  
                    #must open connection first to get reportDir                  
                    self.connection.open()
                    self._configure_logging()
                else:
                    self._configure_logging()
                    self.connection.open()
            finally:
                self.opening = False        
                        
            self._isopen = True
        
        
                 
    def close(self):
        Adapter.close(self)
        if hasattr(self.connection, 'opened') and self.connection.opened:
            self.connection.close()        

    def check_size(self, serializerInst, logfilename, is_result=False, size_limit = None):
        """Check if transmission size exceeds limit"""
        if size_limit is None:
            size_limit = self.max_transmit_data
        totalsize = len(serializerInst.serializedObject)
        if totalsize > self.max_transmit_data:
            exmessage = 'Excessive data (%i bytes) transmitted.  See help at %s' % (totalsize, 'http://docs.picloud.com')
            if hasattr(serializerInst,'str_debug_report'):
                exmessage += ' Snapshot of what you attempted to %s:\n' % ('return' if is_result else 'send')
                serializerInst.set_report_minsize(totalsize/3)
                exmessage += serializerInst.str_debug_report(hideHeader=True)
            else:
                exmessage += '\n'
            exmessage += 'Cloud has produced this error as %s too large of an object.\n' %('your job is returning' if is_result else 'you are sending')
            if hasattr(self,'str_debug_report'):
                if is_result:
                    exmessage += 'The above snapshot describes the data that would be returned by this job.\n'
                else:
                    exmessage += 'The above snapshot describes the data that must be transmitted to execute your PiCloud call.\n'
            else:
                exmessage += '\n To see data snapshots, use _fast_serialization = 0 (default)'
                if not cc.transport_configurable('running_on_cloud',default=False):
                    exmessage += ' and enable serialize_debugging in cloudconf.py\n'
                else:
                    exmessage += '\n'
            if is_result:
                exmessage += 'You cannot return more than %s MB from a job' % (size_limit / 1000000)
            elif cc.transport_configurable('running_on_cloud',default=False):
                exmessage += 'You cannot send more than %s MB per job argument via cloud.call/map' % (size_limit / 1000000)
            else:
                exmessage += 'If you decide that you actually need to send this much data, increase max_transmit_data in cloudconf.py (max 16 MB)\n'
             
            if logfilename:
                exmessage+= 'See entire serialization snapshot at ' + logfilename
            elif not self.cloud.running_on_cloud():
                exmessage+='Please enable serialize_logging in cloudconf.py to see more detail'
            raise CloudException(exmessage)
    
    def _cloud_serialize_helper(self, func, arg_serialization_level, args, argnames=[], logprefix="", 
                                coerce_cnt=None, os_env_vars=[]):
        """Returns a serialization stream which produces a serialized function, then its arguments in order
        Also returns name of logfile and any counter associated with it
        """
        baselogname =  logprefix

        if isinstance(func,partial):
            f = func.func
        else:
            f = func
        
        if f:
            if hasattr(f,'__module__'):            
                baselogname+=(f.__module__ if f.__module__ else '__main__') +'.'
            if hasattr(f,'__name__'):
                baselogname+=f.__name__        
        
        acname = None                
      
        sargs = []
        
        if coerce_cnt != None:
            cnt = coerce_cnt
        else:
            cnt = self._report.update_counter(baselogname) if self._report else 0

        def generate_stream():
            """Helper generator that produces the stream"""
            #serialize function:
            if f:
                sfunc = self.getserializer(0)(func)
                sfunc.set_os_env_vars(os_env_vars)
                try:
                    sfunc.run_serialization(self.min_size_to_save)
                finally:        
                    if self.serializeLogging:
                        logname = baselogname + '.%sfunc'
                        acname = self._report.save_report(sfunc, logname, cnt)
                    else:
                        acname = ""  
                
                self.check_size(sfunc,acname) 
                yield sfunc
            
            #arguments
            if args:
                argSerializer = self.getserializer(arg_serialization_level)
                for obj, name in izip(args, argnames):
                    #TODO: Policy change here
                    serializerI = argSerializer(obj)
                
                    try:
                        serializerI.run_serialization(self.min_size_to_save)
                    finally:
                        if self.serializeLogging:                        
                            logname = baselogname + '.%s' + name
                            acname = self._report.save_report(serializerI, logname, cnt)  
                        else:
                            acname = ""
                
                    self.check_size(serializerI,acname) 
                
                    yield serializerI                                        
        return generate_stream(), baselogname, cnt

        
    
    def cloud_serialize(self, func, arg_serialization_level, args, argnames=[], logprefix="", 
                        coerce_cnt=None, os_env_vars=[]):
        """Return serialized_func, list of serialized_args
        Will save func and args to files.
        """
        
        serialize_stream, logname, cnt = self._cloud_serialize_helper(func, arg_serialization_level, 
                                                                      args, argnames, logprefix, 
                                                                      coerce_cnt, os_env_vars)
        if func:
            sfunc = serialize_stream.next().serializedObject
        else:
            sfunc = None        
        if args:
            sargs = imap(lambda sarg: sarg.serializedObject, serialize_stream) #stream
        else:
            sargs = None
                
        return sfunc, sargs, logname, cnt
    
    
    def map_reduce_job(self, mapper_func, reducer_func, bigdata_file):
        
        # serialize the the args above into params
        smapper, _, _, _ = self.cloud_serialize(mapper_func, 0, [])
        sreducer, _, _, _ = self.cloud_serialize(reducer_func, 0, [])
        
        params = {}
        params['mapper_func'] = smapper
        params['reducer_func'] = sreducer
        params['bigdata_file'] = bigdata_file
        
        return self.connection.add_map_reduce_job(params)

    
    def job_call(self, params, func, args, kwargs):
        os_env_vars = params.pop('os_env_vars', None)        
        sfunc, sargs, logprefix, logcnt =  self.cloud_serialize(func, params['fast_serialization'], 
                                                                [args, kwargs], ['args', 'kwargs'], 'call.',
                                                                os_env_vars=os_env_vars) 
        
        params['func'] = sfunc
        params['args'] = sargs.next()
        params['kwargs'] = sargs.next()

        return self.connection.job_add(params=params, logdata = (logprefix, self._report.pid if self._report else 0, logcnt))
    
    def jobs_join(self, jids, timeout = None):
        return self.connection.jobs_join(jids, timeout)
    
    def jobs_map(self, params, func, mapargs, mapkwargs = None):
        """Treat each element of mapargs as func(*element)
        Mapkwargs implementation is a bit inefficient in that it can call two 
        module_add/check. however, this will be a very rare case"""
        
        mapargnames = imap(lambda x: 'jobarg_' + str(x),count(1)) #infinite stream generates jobarg_ for logging
        mapkwargnames = imap(lambda x: 'jobkwarg_' + str(x),count(1)) #infinite stream generates jobarg_ for logging                    
        
        os_env_vars = params.pop('os_env_vars', None)
        sfunc, sargs, logname, logcnt =  self.cloud_serialize(func, params['fast_serialization'], 
                                                              mapargs, mapargnames, 'map.', 
                                                              os_env_vars=os_env_vars)
        #handle kwargs if present
        if mapkwargs:
            _, skwargs, _, _ = self.cloud_serialize(None, params['fast_serialization'], mapkwargs, 
                                                    mapkwargnames, logname, coerce_cnt = logcnt)
        else:
            skwargs = None        

        params['func'] = sfunc
        
        if self.isSlave: #fixme: Find a way to pass stream through to master
            sargs = list(sargs)
            
        return self.connection.jobs_map(params=params, mapargs = sargs, mapkwargs = skwargs,
                                       logdata = (logname, self._report.pid if self._report else 1, logcnt))
        
    def jobs_result(self, jids, by_jid):
        """Returns serialized result; higher level deals with it"""
        return self.connection.jobs_result(jids=jids, by_jid=by_jid)
    
    def jobs_kill(self, jids):
        return self.connection.jobs_kill(jids=jids)
    
    def jobs_delete(self, jids):
        return self.connection.jobs_delete(jids=jids)        
    
    def jobs_info(self, jids, info_requested):
        return self.connection.jobs_info(jids=jids, info_requested=info_requested)
   
    def is_simulated(self):        
        return self.connection.is_simulated()
    
    def needs_restart(self, **kwargs):        
        return self.connection.needs_restart(**kwargs)

class DependencyAdapter(SerializingAdapter):
    """
    The DependencyAdapter is a SerializingAdapter capable of sending modules
    to PiCloud
    """
    

    # manages file dependencies:
    dependencyManager = None


    c = """Should a more aggressive, less optimal, module dependency analysis be used?  This is always used on stdin"""
    aggressiveModuleSearch = \
        cc.transport_configurable('aggressive_module_search',default=False,comment=c, hidden=True)

    c = """Set to False to disable automatic dependency transfer. This will not affect dependencies already sent to PiCloud."""
    automaticDependencyTransfer = \
        cc.transport_configurable('automatic_dependency_transfer',default=True,comment=c, hidden=False)


    def _check_forced_mods(self):
        """find things that may be imported via import a.b at commandline
        Set __main__.___pyc_forcedImports__ to that list
        """
        marked = set()        
        main = (sys.modules['__main__'])  
        marked.add(main)        
        def recurse(testmod, name): #name is tuple version of testmod
            for key, val in testmod.__dict__.items():                
                if not name: #inspecting main module -- use val name
                    if isinstance(val,types.ModuleType):
                        item = val.__name__
                        if item in ['cloud','__builtin__']:
                            continue
                        tst = tuple(item.split('.'))
                        
                    else:
                        continue
                else: #child module cannot be renamed inside a parent
                    item = key
                    tst = name + (item,) 
                if item in marked: 
                    continue                
                tstp = '.'.join(tst)                
                themod = sys.modules.get(tstp)
                if not themod:
                    continue
                
                self._create_dependency_manager()
                if self.dependencyManager:
                    self.dependencyManager.inject_module(themod)  
                marked.add(item)
                if len(tst) > 1: #this is a forced import:
                    if not hasattr(main,'___pyc_forcedImports__'):
                        main.___pyc_forcedImports__ = set()
                    main.___pyc_forcedImports__.add(themod)
                recurse(themod,tst)                                    
        with self.modDepLock:
            recurse(main,tuple())                

    
    def _serialize_dep_handler(self, sarg):
        """Used by cloud_serialize imap on serialized args"""
        
        self._create_dependency_manager()
        if self.dependencyManager:
            with self.modDepLock:
                for moddep in sarg.get_module_dependencies():
                    self.dependencyManager.inject_module(moddep)
        return sarg.serializedObject
        
    
    def cloud_serialize(self, func, arg_serialization_level, args, argnames=[], logprefix="", 
                        coerce_cnt=None, os_env_vars=[]):
        """
        Similar to the normal cloud_serialize, except tracks dependencies
        """
        
        try:
            self._create_dependency_manager()
                           
            main = (sys.modules['__main__'])
            main_file = getattr(main, '__file__', None)
            # TODO: ipython support?
            force_aggressive = not main_file or main_file == '<stdin>'
            
            if serialization.cloudpickle.useForcedImports and \
                (self.aggressiveModuleSearch or force_aggressive):
                self._check_forced_mods()      
    
            if not self.automaticDependencyTransfer:
                return SerializingAdapter.cloud_serialize(self, func, arg_serialization_level, 
                                                          args, argnames, logprefix, coerce_cnt,
                                                          os_env_vars)   
            
            serialize_stream, logname, cnt = self._cloud_serialize_helper(func, arg_serialization_level, 
                                                                          args, argnames, logprefix, 
                                                                          coerce_cnt, os_env_vars)
                    
            if func:
                sfunc = serialize_stream.next()
                outfunc = self._serialize_dep_handler(sfunc)
            else:
                func = None
                outfunc = None
                        
            if args:
                outargs = imap(lambda sarg: self._serialize_dep_handler(sarg), serialize_stream)
            else:
                outargs = None
    
            return outfunc, outargs, logname, cnt
        
        except Exception, e: # report exceptions
            
            
            try:
                from .network import HttpConnection, cloudLog as conLog
                import platform
                import traceback
                
                if isinstance(self.connection, HttpConnection):
                    self.connection.send_request('report/python_error/', 
                                                 {'exception' : str(e), 
                                                  'stacktrace' : ''.join(traceback.format_stack()),
                                                  'traceback' : ''.join(traceback.format_tb(sys.exc_info()[2])),                                              
                                                  'hostname': platform.node(),
                                                  'language_version': platform.python_version(),
                                                  'language_implementation': platform.python_implementation(),
                                                  'platform': platform.platform(),
                                                  'architecture': platform.machine(),
                                                  'processor': platform.processor(),
                                                  'pyexe_build' : platform.architecture()[0]
                                                      }, logfunc=conLog.debug )                
                
            except Exception:
                pass # absolutely do not let an exception corrupt user seeing the true error                
            raise # propagate old exception 

    def dep_snapshot(self):
        """This function, called downstream from the network connection object, 
        deals with checking for new modules and sending them over the network
        
        It does not return anything"""
        
        self._create_dependency_manager()
        if not self.dependencyManager:
            return
        
        with self.modDepLock:
            #lock is held entire time as small chance webserver will raise CloudException            
            deps, new_snapshot = self.dependencyManager.get_updated_snapshot()        
            
            if deps:
            #if False:
                cloudLog.debug('New Dependencies %s' % str(deps))
                #modules are a tuple of (modname, timestamp, archvie)
                modules = self.connection.modules_check(deps)
                if modules:
                    from ..transport.codedependency import FilePackager
                    f = FilePackager(map(lambda module: (module[0], module[2]), modules), self.dependencyManager)
                    tarball = f.get_tarball()
                    cloudLog.info('FileTransfer: Transferring %s' % modules)
                    self.connection.modules_add(modules, tarball)
            
            #if no exception raised, commit snapshot
            self.dependencyManager.commit_snapshot(new_snapshot)
    
    def _create_dependency_manager(self, whitelist=None):
        
        if not self.automaticDependencyTransfer:        
            return
        elif not self.dependencyManager:
            from ..transport import DependencyManager
            ignoreList = self.connection.packages_list()
            self.dependencyManager = DependencyManager(excludes=ignoreList, whitelist=whitelist)
    
    def open(self):                
        with self.openLock:
            if self.opening or self.opened:
                return
            SerializingAdapter.open(self)
            if self._cloud.running_on_cloud(): 
                # within cloud, we have all needed dependencies so no need to transfer any
                self.automaticDependencyTransfer = False            
            self.modDepLock = threading.Lock()            
            self._isopen = True
