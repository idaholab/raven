"""
This implements the multiprocessing adapter.  Much is added to the baseline Multiprocessing.Pool

Note: This code is unfortunately a bit of a mess due to the sheer number of extensions
Ideally, it would be rewritten one day

Clean-up notes:
    -Seperate classes into various modules
    -Clear-up some duplicate code (killing is very messy esp w/ process respawn 
        - borrow code from pool - 2.7)
        - clear_results is rediculous -- should interface w/ handle_results (in pool)
    -Consider no longer subclassing from pool -- only leads to more confusion w/ logging, etc
    -Seperate open into variety of functions
    

Copyright (c) 2011 `PiCloud, Inc. <http://www.picloud.com>`_.  All rights reserved.

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
import os
import inspect
import time
import types
import socket
import traceback
import threading
import itertools
import Queue
import traceback
import cProfile
import signal
from multiprocessing import cpu_count, current_process
from multiprocessing.managers import BaseManager, State, Server, Token, AutoProxy, BaseProxy, dispatch
from multiprocessing.pool import Pool
import multiprocessing.pool as pool
from multiprocessing.util import Finalize, debug, _run_finalizers
#todo: Set debug to use cloudLog?

from ..cloud import CloudException, CloudTimeoutError
from ..serialization import deserialize
from ..util.xrange_helper import maybe_xrange_iter
from .connection import CloudConnection
from .adapter import SerializingAdapter
from .. import cloudconfig as cc
import logging
cloudLog = logging.getLogger('Cloud.mp') 

use_profiler = False   #For internal testing use

#status:
#Note: No assigned as it makes no sense in this context
status_waiting = 'waiting'  #waiting for dep to be satisfied
status_new = 'queued'  #new job in queue  (should this be queued??)
status_processing = 'processing' #processing
status_done = 'done' #job done
status_error = 'error' #job messed up
status_stalled = 'stalled' #job fails due to dependency failing
status_killed = 'killed' #job explicitly killed

ran_statuses = [status_done, status_error, status_stalled, status_killed]
error_statuses = [status_error, status_stalled, status_killed]

#constants
start_processing_flag = True
sleep_task_flag = True

#timer taken from timeit
if sys.platform == "win32":
    # On Windows, the best timer is time.clock()
    default_timer = time.clock
else:
    # On most other platforms the best timer is time.time()
    default_timer = time.time


class KilledException(Exception):
    """Exception used when job killed"""
    pass

#
# Code run by worker processes
# Modified from base mp to notify when starting to process and receive backoff messages
#
def cloud_worker(inqueue, outqueue, cloudHooks, mypid = 0, redirect_job_output=True, 
                 serialize_debugging = False, serialize_logging = False,
                 api_key=None, api_secretkey=None):
    put = outqueue.put
    get = inqueue.get
    if hasattr(inqueue, '_writer'):
        inqueue._writer.close()
        outqueue._reader.close()

    #bind last keys:
    from .. import cloudinterface
    cloudinterface.last_api_key = api_key
    cloudinterface.last_api_secretkey = api_secretkey
    
    new_cloud = patch_cloud(cloudHooks, mypid)
    adapter = new_cloud.adapter
    report = adapter.report   
    
    if not report and redirect_job_output:  #disable redirection!
        cloudLog.warn('Cannot redirect Job Output due to report construction failing. Falling back to console output')
        redirect_job_output = False 
    
    #options:
    adapter.serializeDebugging = serialize_debugging
    adapter.serializeLogging = serialize_logging

    base_stdout = sys.stdout
    base_stderr = sys.stderr
        
    #print 'worker online'
    
    while 1:
        try:
            task = get()
            #print 'worker got', task
        except (EOFError, IOError):
            debug('worker got EOFError or IOError -- exiting')
            break

        if task is None:
            debug('worker got sentinel -- exiting')
            break
        
        if task is sleep_task_flag: #help killing
            time.sleep(2.0)
            continue

        job, func, args, kwds, result_serialization_level, logname, logpid, logcnt = task
        
        if func:
            func = deserialize(func)
        if args:
            args = deserialize(args)
        if kwds:
            kwds = deserialize(kwds)
        
        put ((job, mypid, start_processing_flag)) #None as result indicates this is started
                        
        startTime = default_timer() #in case exception triggers early
        try:                                                                        
            if redirect_job_output:  #need a proxy file writer to redirect!
                stdout = report.open_report_file(logname, '.stdout', logcnt, logpid)
                stderr = report.open_report_file(logname, '.stderr', logcnt, logpid) 
                
                sys.stdout = stdout 
                sys.stderr = stderr
            
            funcres = None
            
            # catch SIGINT (kill commands) and simply raise an exception
            # so that we get access to the traceback            
            # Note: os.kill() doesn't work on win32 for python < 2.7, preventing signaling
            if sys.platform != 'win32' or sys.version_info >= (2.7):
                signal.signal(signal.SIGINT, sigterm_handler) 
            
            if use_profiler and report:
                pfname = report.get_report_file(logname, '.profile', logcnt, logpid)
                #print pfname
                
                prof = cProfile.Profile()
                startTime = default_timer()
                funcres = prof.runcall(func, *args, **kwds)
                prof.dump_stats(pfname)
            else:
                startTime = default_timer()
                funcres = func(*args, **kwds)  
            endTime = default_timer()     
            
            sfuncres = adapter.getserializer(result_serialization_level)(funcres)  
            try:
                sfuncres.run_serialization()          
            finally:
                if adapter.serializeLogging and report:                    
                    acname = report.save_report(sfuncres, logname, logcnt, logpid)
                else:
                    acname = ''
            #log here + checksize
            adapter.check_size(sfuncres, acname)
                        
            result = (True, sfuncres.serializedObject)
        except (SystemExit, Exception), e: 
            endTime = default_timer()
            tb = traceback.format_exc()       
            excp_type = type(e)
            # hack: Sometimes custom tyoes aren't serializable. So just fallback to Exception for safety 
            if excp_type.__module__ not in ['exceptions', 'cloud.transport.local']:
                excp_type = Exception                
            result = (excp_type, tb)
        finally:
            # set signal handler back to default
            if sys.platform != 'win32' or sys.version_info >= (2.7):
                signal.signal(signal.SIGINT, signal.SIG_DFL)                      
            
            if redirect_job_output:                
                if sys.stdout != base_stdout:
                    sys.stdout.close()
                    sys.stdout = base_stdout    
                if sys.stdout != base_stderr:
                    sys.stderr.close()
                    sys.stderr = base_stderr
            
        put((job, endTime - startTime, result))

def sigterm_handler(signum, frame):
    """Handler for preserving tracebacks on cloud.kill"""
    cloudLog.info('Signal handler called with signal %s', signum)    
    signal.signal(signal.SIGINT, signal.SIG_DFL)  #back to default                       
    raise KilledException('Job terminated by user')


handlerInit = False
def setupLogHandlers():
    """Debug statements to bind multiprocessing to cloudlog"""        
    
    global handlerInit
    if handlerInit:
        return
    handlers = cloudLog.parent.handlers #found in main cloudlog
    
    from multiprocessing import get_logger
    
    mplogger = get_logger()
   
    mplogger.setLevel(3) #bind to cloudlog
       
    for handler in handlers:
        mplogger.addHandler(handler)
    
    handlerInit = True
        
class CloudJob(object):
    """CloudJob: 
        Modified version of ApplyResult.
    """
    killMe = False #set to true to self-destruct once processing notification comes in
    deleted = False #used only for killing deleted jobs - if set, do not flush result back 
    _runningpid = None
    _runtime = 0.0 #default runtime
    
    job_counter = itertools.count() #class counter
    job_counter.next() #coerce to start at 2

    
    def __init__(self, pool, func, args, kwds, func_name, label, priority,
                 result_serialization_level, logname="", logpid = 1, logcnt=0):
        self._jid = self.job_counter.next()
        self._pool = pool        
        pool.cache[self._jid] = self
        self._status = status_new
        self._value = None
        self.func = func
        self.args = args
        self.kwds = kwds
        self.func_name = func_name
        self.label = label
        self.priority = priority
        self.result_serialization_level = result_serialization_level
        self.logname = logname
        self.logpid = logpid
        self.logcnt = logcnt        

    @property
    def ready(self):        
        return self._status in ran_statuses 
    
    @property
    def jid(self):
        return self._jid
    
    @property
    def status(self):
        return self._status
    
    @property
    def pid(self):
        return self._pid
    
    @property
    def result(self):
        return self._value
    
    @property
    def runtime(self):
        st = getattr(self, '_start_time', None)
        if st:
            return default_timer() - st
        else:
            return self._runtime
    
    def _aborted(self, abort_status, result = None):
        if hasattr(self, '_start_time'):
            endTime = time.time()
            self._runtime = endTime - self._start_time
            del self._start_time        
        self._status = abort_status
        self._value = result        
        self._pool.notify_job_done(self._jid)
        try:
            self._pool._task_sem.release()
        except ValueError, e: 
            cloudLog.debug('Semaphore release triggered %s', str(e))
        del self._pool.cache[self._jid]
    
    def _set(self, extra_data, sobj):
        """Update with some arbitrary extra_data and some representation of a returned object"""
        if sobj is start_processing_flag: #flag for processing
            #print 'set to processing %s k=? %s' % (self.status, self.killMe)
            if self.status == status_killed: #killed?
                return             
            self._start_time = default_timer()  
            self._pid = extra_data             
            
            self._status = status_processing
            
            
            try:
                self._pool._task_sem.release()
            except ValueError, e: 
                cloudLog.debug('Semaphore release triggered %s', str(e))

            
            if self.killMe: #important: this check must come AFTER processing set
                cloudLog.debug("Auto-Killing job after started processing. jid=%d, label=%s, func=%s, " % (self.jid, self.label, self.func_name))
                self._pool._do_job_kill(self)
            else:
                cloudLog.debug("Processing job. jid=%d, label=%s, func=%s, " % (self.jid, self.label, self.func_name))            
            return                               
        
        if self.status in ran_statuses: #killed?
            return
        
        self._runtime = extra_data
        del self._start_time
        success, self._value = sobj        
        if success == True:
            self._status = status_done
            self._value = self._value
        elif self._status != status_killed:
            if success == KilledException:
                self._status = status_killed
            else:
                self._status = status_error
            

        cloudLog.debug("Finished job. jid=%d, label=%s, func=%s, status=%s, runtime=%g" % (self.jid, self.label, self.func_name, self.status,  self.runtime))            
        
        self._pool.notify_job_done(self._jid)
        
        try: #might not exist if killing
            del self._pool.cache[self._jid]
        except:
            pass
        
    def __str__(self):
        return 'CloudJob <jid: '+str(self.jid)+', func_name: '+self.func_name + \
                         ', label: '+str(self.label)+', status:'+self.status + '>'

class ThreadedServer(Server):
    """With self descruct ability"""
    
    destroy_self = False
    close_at = None #if set to a time, break at this time
    close_wait_time = 0.1 #time to wait for closing
     
    def serve_forever(self):
        '''
        Run the server forever
        '''
        current_process()._manager_server = self
        try:
            try:
                while 1:
                    try:
                        c = self.listener.accept()
                    except socket.timeout:
                        cloudLog.debug('Timeout triggered')
                    except (OSError, IOError):
                        continue
                    if self.destroy_self:
                        if self.close_at:
                            if time.time() > self.close_at: 
                                break
                        else:                                                        
                            self.close_at = time.time() + self.close_wait_time
                            #massive hack: allow accept to abort with a timeout
                            self.listener._listener._socket.settimeout(self.close_wait_time)                            
                            cloudLog.debug('Threaded server will shut down at %s', self.close_at)
                    t = threading.Thread(target=self.handle_request, args=(c,), name='handle_request')
                    t.daemon = True
                    t.start()
            except (KeyboardInterrupt, SystemExit):
                pass
        finally:
            self.stop = 999
            self.listener.close()
            cloudLog.info('ThreadedServer shut down. destroy_self was %s' % self.destroy_self)
           
class MPManager(BaseManager):
    """Modified to run in same thread as initializer
    IMPORTANT: Because this binds the instance to the class, only one of these objects can ever exist"""
    server = None
    
    waitCV = None
    
    myInstance = None #only once instance allowed
    
    _Server = ThreadedServer

    @classmethod
    def __new__(cls, *args, **kwds):
        """Singleton -- Do not allow multiple initializations"""
        if cls.myInstance:
            return None        
        else: 
            instance =  BaseManager.__new__(*args, **kwds)
            cls.myInstance = instance
            return instance

    @classmethod
    def _run_server_nosend(cls, registry, address, authkey, serializer):
        '''
        Create a server, report its address and run it
        '''
        
        # create server
        cls.server = cls._Server(registry, address, authkey, serializer)

        # run the manager
        cloudLog.debug('Mpmanager serving at %r', cls.server.address)
               
        with cls.waitCV:
            cls.waitCV.notify()
        #print 'try start server..'
        cls.server.serve_forever() 
           
    
    def start_threaded(self):
        """Launch this as a threaded application"""
        assert self._state.value == State.INITIAL

        self.__class__.waitCV = threading.Condition()
        
        # spawn thread which runs the server
        self._thread = threading.Thread(
            target=type(self)._run_server_nosend,
            name = 'run_server_nosend',
            args=(self._registry, self._address, self._authkey,
                  self._serializer)
            )
        self._thread.daemon = True

        
        with self.__class__.waitCV:
            self._thread.start()
            self.__class__.waitCV.wait()

        ident = self._thread.ident if hasattr(self._thread,'ident') else '' #python 2.5 lacks ident
        cloudLog.debug('Manager started with thread ident %s' % ident)
        self._thread.name = type(self).__name__  + '-' + str(ident)
        self._address = self.__class__.server.address  
        
        # register a finalizer (TODO):
        self._state.value = State.STARTED
        
    def close(self):   
        """
        Closing is tricky, due to risk of receiving on a dead connection
        Strategy we use is to force mpmanager to wait a little before fully closing"""
        
        cloudLog.debug('Starting pool close')
        self.__class__.server.destroy_self = True            
        
        try:
            conn = self._Client(self._address) #do not attempt to receive data as connection may be dead
        except (IOError, EOFError): #all good
            if not self._thread.is_alive():                
                cloudLog.debug('MpManager successfully shut down early')
            else:    
                cloudLog.warning('MPManager Thread alive but cannot bind to %s', self._address, exc_info=True)                    
        except Exception:
            cloudLog.debug('Cloud not bind to %s.' % self._address, exc_info=True)
        else: 
            try:                
                cloudLog.debug('Client to server shutdown %s' % self._address)
                conn.send(('#RETURN', None)) #garbage to force server to self-descruct
                cloudLog.debug('Sent #RETURN')
            finally:                
                conn.close()            
        
        cloudLog.debug('Finalizers start')
        _run_finalizers()
        cloudLog.debug('Finalizers end -- killing manager')
        
            
        #block until mpmanager down
        self._thread.join(self.__class__.server.close_wait_time + 0.07)
        if self._thread.is_alive():
            cloudLog.error('Failed to shut down mp manager!')
        else:
            cloudLog.debug('mp manager shut down successfully')
            
        #reset!
        self.__class__.myInstance = None
        try:
            del BaseProxy._address_to_local[self._address]
        except KeyError: #may already be removed by finalizer
            pass
        current_process()._tempdir = None 
            
    
    @classmethod
    def register(cls, typeid, callable=None, proxytype=None, exposed=None,
                 method_to_typeid=None, create_method=True):
        '''
        Register a typeid with the manager type
        Slight modification to in the method created
        '''
        if '_registry' not in cls.__dict__:
            cls._registry = cls._registry.copy()

        if proxytype is None:
            proxytype = AutoProxy

        exposed = exposed or getattr(proxytype, '_exposed_', None)

        method_to_typeid = method_to_typeid or \
                           getattr(proxytype, '_method_to_typeid_', None)

        if method_to_typeid:
            for key, value in method_to_typeid.items():
                assert type(key) is str, '%r is not a string' % key
                assert type(value) is str, '%r is not a string' % value

        cls._registry[typeid] = (
            callable, exposed, method_to_typeid, proxytype
            )

        if create_method:
            def temp(self, *args, **kwds):
                debug('requesting creation of a shared %r object', typeid)

                oid, exp = self.__class__.server.create(None,typeid, *args, **kwds)
                token = Token(typeid, self._address, oid)
                proxy = proxytype(
                    token, self._serializer, manager=self,
                    authkey=self._authkey, exposed=exp
                    )
                conn = self._Client(token.address, authkey=self._authkey)
                dispatch(conn, None, 'decref', (token.id,))
                return proxy
            temp.__name__ = typeid
            setattr(cls, typeid, temp)        

class MPConnectionHook(object):
    """Wrapper object to make calls to MPConnection from subprocesses
    """
    
    def __init__(self, cloudcon, cloudmodule, reportDir):       
        self.cloudcon = cloudcon
        self.cloudmodule = cloudmodule
        self.reportDir = reportDir
    
    def open(self):
        """Open master if it isn't open"""
        if not self.cloudcon.opened:
            self.cloudcon.open()
        if self.cloudcon.adapter.report:
            self.reportDir = self.cloudcon.adapter.report.logPath
        self.opened = self.cloudcon.opened
    
    def close(self):
        """cannot close the proxy"""
        pass
    
    def get_cloud_module(self):
        return self.cloudmodule
    
    def get_report_dir(self):
        return self.reportDir
    
    def job_add(self, params, logdata = None):
        return self.cloudcon.job_add(params, logdata)    
        
    def jobs_join(self, jids, timeout):
        """jobs join is not supported here as the server is not multithreaded"""
        return False                

    def jobs_map(self, params, mapargs, mapkwargs = None, logdata = None):
        return self.cloudcon.jobs_map(params, mapargs, mapkwargs, logdata)
    
    def jobs_result(self, jids, by_jid):
        return self.cloudcon.jobs_result(jids, by_jid)
    
    def jobs_kill(self, jids):
        return self.cloudcon.jobs_kill(jids)
    
    def jobs_delete(self, jids):
        return self.cloudcon.jobs_delete(jids)
    
    def jobs_info(self, jids, info_requested):
        return self.cloudcon.jobs_info(jids, info_requested)
    
    def is_simulated(self):
        return self.cloudcon.is_simulated()
    
    def needs_restart(self, **kwargs):
        return self.cloudcon.needs_restart(**kwargs)
    
    def send_request(self, url, post_values, get_values=None, logfunc=cloudLog.info):
        return self.cloudcon.send_request(url, post_values, get_values, logfunc)          
    
    def connection_info(self):
        return self.cloudcon.connection_info()
    
    def force_adapter_report(self):
        """
         Should the SerializationReport for the SerializationAdapter be coerced to be instantiated?
        """
        return self.cloudcon.force_adapter_report()
    
    def report_name(self):
        return self.cloudcon.report_name()       
    

def patch_cloud(cloudObjs, myPid):
    """Run on subprocesses to patch cloud with a list of cloudproxies
    Returns the primary cloud, constructed from the first cloudproxy"""
    from ..cloud import Cloud
    from .. import cloudinterface as ci
    
    report = None
    
    baseCl = None
    
    for cloudObj in cloudObjs:
        
        #construct cloud    
        sa = SerializingAdapter(cloudObj, isSlave = True)   
        reportDir = cloudObj.get_report_dir()
        if reportDir:
            sa.init_report()
            report = sa.report
            report.logPath = reportDir
            report.pid = myPid  #restore when supported 
            
        cl = Cloud(sa)
        
        modstr = cloudObj.get_cloud_module()
        __import__(modstr)
        mod = sys.modules[modstr]
            
        ci._bindcloud(mod, cl, 'proxy', immutable=True)
        if not baseCl:
            baseCl = cl
        
    return baseCl

def tail_file(filename, max_bytes=64000):
    """
    Returns the last max_bytes of a file. The storage is used to
    access the passed in filename.
    """
    
    size = os.path.getsize(filename)
    pos = max(0, size - max_bytes)
    
    f = file(filename)
    f.seek(pos)
    bytes = f.read()
    f.close()
    
    return bytes    
 
class MPConnection(CloudConnection, Pool):
    inKill = False
        
    #CONFIGURATION:
    c = """Number of subprocesses that cloud multiprocessing should utilize.
    Beware of using too low of a number of subprocesses, as deadlock can occur. 
    If set to 0, this will be set to the number of cores available on this system."""
    num_procs =  cc.mp_configurable('num_procs', default=8, comment=c)
    
    c = """Allow at most this factor*number subprocesses to be removed from the 
    internal priority queue and placed on the pipe."""
    io_buffer_factor =  cc.mp_configurable('io_buffer_factor', default=2.5, comment=c, hidden=True)
    
    c = """If set to true, job stdout and stderr will be redirected to files inside 
    the serialize_logging_path.  This option simulates PiCloud behavior and must be
    set to true for ``cloud.info`` to be able to request stdout or stderr information 
    about jobs.
    If set to false, job stdout and stderr will be the same as the parent process."""
    redirect_job_output = cc.mp_configurable('redirect_job_output', default=True, comment=c)
        
    c = """Maxmimum number of jobs that cloudmp should store.  Set to 0 for no limit.  
    This option only applies to cloud.mp and the cloud simulator.
    PiCloud recommends that any limit stays above 1024."""
    mp_cache_size = cc.mp_configurable('mp_cache_size_limit', default=0, comment=c)
            
    support_join = cc.mp_configurable('support_join', default = True, hidden=True,
                                     comment = 'Are high speed joins supported?')
    
    
        
    def __init__(self):
        self.openLock = threading.RLock()


    @property
    def pool(self):
        return self._pool
    
    def connection_info(self):
        dict = CloudConnection.connection_info(self)
        dict['connection_type'] = 'multiprocessing'        
        dict['num_procs'] = self.num_procs
        return dict    
    
    @property
    def cache(self):
        return self._cache
    
    def open(self):
        with self.openLock:
            
            if self._isopen:
                return            
            
            MPManager.register('ConnectionHook',MPConnectionHook) #force registration
            
            CloudConnection.open(self)        

            if self.mp_cache_size == 0:
                self._adapter.cloud.job_cache_size = 0
            else:
                self._adapter.cloud.job_cache_size = self.mp_cache_size
            self._adapter.cloud.result_cache_size = 0 #no limit in multiprocessing
                   
            #pool related
            self._setup_queues()
            
            if hasattr(Queue,'PriorityQueue'): #python 2.6+
                self._taskqueue = Queue.PriorityQueue()
            else:
                self._taskqueue = Queue.Queue()
            self._cache = {}
            self._state = pool.RUN
                    
            self.jobDeps = {}  #value is list of jobs that key depends on
            self.jobInvDeps = {} #value is list of jobs that depend on key
    
            #proxy objects
                    
            #Don't spawn more than one copy!
            if MPManager.myInstance:
            
                cloudmanager = MPManager.myInstance
            else:
                cloudmanager = MPManager()         
                cloudmanager.start_threaded()
            self.cloud_manager = cloudmanager
                
            #cloud-hooks. We must bind cloud and cloud.mp modules correctly
            self.cloudHooks = []        
            modnames = ['cloud', 'cloud.mp']
            
            for modname in modnames:
                try:
                    cloudmod =  sys.modules[modname]
                except KeyError:
                    cloudLog.error('Cannot find %s in sys.modules' % modname)
                else:
                    cloud = getattr(cloudmod,'__cloud')
                    adapter = cloud.adapter
                    conn = adapter.connection                                
                                        
                    if adapter.report:
                        reportDir = adapter.report.logPath
                    else:
                        reportDir = None
                    
                    chook = cloudmanager.ConnectionHook(conn, modname, reportDir)                    
                    
                    #first cloudhook is 'primary' -- i.e. this object
                    if conn == self:
                        self.cloudHooks.insert(0, chook)
                    else:
                        self.cloudHooks.append(chook)                
                    
            if not self.num_procs:
                try:
                    self.num_procs = cpu_count()
                except NotImplementedError:
                    self.num_procs = 1
    
            self._pool = []
            self.nextpid = 2
            
            for _ in xrange(self.num_procs):
                self._genprocess()        
    
            self._task_sem = threading.BoundedSemaphore(int(self.io_buffer_factor*self.num_procs))
    
            self._task_handler = threading.Thread(
                target=MPConnection._cloud_handle_tasks,
                name = '_cloud_handle_tasks',
                args=(self._taskqueue, self._task_sem, self._quick_put, self._outqueue, self._pool)
                )
            self._task_handler.daemon = True
            self._task_handler._state = pool.RUN
            
            self._task_handler.start()                
    
            self._result_handler = threading.Thread(
                target=Pool._handle_results,
                name = '_cloud_handle_results',
                args=(self._outqueue, self._quick_get, self._cache)
                )
            self._result_handler.daemon = True
            self._result_handler._state = pool.RUN
            self._result_handler.start()
            
            self._process_monitor_cv = threading.Condition()
            self._process_monitor = threading.Thread(
                target=self._monitor_pool,
                name = 'monitor_pool',
                args=()
                )            
            self._process_monitor.daemon = True
            self._process_monitor._state = pool.RUN
            self._process_monitor.start()
            
    
            self._terminate = Finalize(
                self, self._terminate_pool,
                args=(self._taskqueue, self._inqueue, self._outqueue, self._pool,
                      self._process_monitor, self._task_handler, self._result_handler, self._cache),
                exitpriority=15
                )
                
            self._worker_handler = self._process_monitor 

            self.dependency_lock = threading.RLock()
            self.job_kill_lock = threading.RLock()            
            
            #Joining  - A very simple implementation is used:
            #    jobs_join sleeps on join_cv until the jobs are not done
            #    Whenever a job completes, it wakes everyone sleeping on the join_cv 
            # This is not the most efficient system, but with the limited number of jobs 
            # being run, it doesn't matter
            if self.support_join:                
                self.join_cv = threading.Condition()
            else:
                self.join_cv = None
            
            #setupLogHandlers()
            cloudLog.info('Started Cloud multiprocessing with %d subprocesses' % self.num_procs)
            
            
   
    def _genprocess(self):
        from .. import cloudinterface
        w = self.Process(
            target=cloud_worker,
            args=(self._inqueue, self._outqueue, self.cloudHooks,  
                  self.nextpid, self.redirect_job_output,
                  self.adapter.serializeDebugging, self.adapter.serializeLogging,
                  cloudinterface.last_api_key, cloudinterface.last_api_secretkey)
            )
        self._pool.append(w)
        w.name = w.name.replace('Process', 'CloudPoolWorker')
        w.daemon = True
        w.start()
        w.cloudPid = self.nextpid #extra info
        self.nextpid +=1 

    def _kill_job(self, jid):        
        #TODO: We can abort early if we access the pqueue
                            
        job = self._cache.get(jid)
        if not job: #already done
            job = self.get_job_item(jid)  
            return job.status
        job.killMe = True #terminate job when it starts processing
        if job.status in ran_statuses:
            return job.status
        if job.status == status_processing: #important: this check must occur AFTER killMe set!
            self._do_job_kill(job)
        return status_killed #will be killed?               
        
    def _do_job_kill(self, job):
        """Kill process owned by job if necessary"""        
        
        #first we try a SIGINT which may raise an exception and get us out of here
        #not supported on windows when running python versions < 2.7
        if sys.platform != 'win32' or sys.version_info >= (2.7):            
            cloudLog.info('Sending SIGINT to job %d on pid %d. real pid' % (job.jid, job.pid))
            for proc in self._pool:
                if proc.cloudPid == job.pid:                      
                    os.kill(proc.ident, signal.SIGINT)
        
        time.sleep(1.0)
        cloudLog.info('Job is currently %s' % job.status)
        if job.status != status_processing: #sigint worked!
            return
            
        with self.job_kill_lock:            
            
            re_enter = False
            if self.inKill:
                re_enter = True
            elif self._outqueue._wlock:
                self._outqueue._wlock.acquire() #lock down write backs (block corruption) 
            #the write lock also holds up subprocesses on reading        
            killed = False                            
            self.inKill = True
            try:
                timeout = 0.1
                racquire = self._inqueue._rlock.acquire
                cloudLog.info('Initiating shutdown of job %d on pid %d' % (job.jid, job.pid))
                if not re_enter:
                    for _ in xrange(3): #a few tries
                        if racquire(block=True, timeout = timeout): #lock down lock job process reads
                            break
                        cloudLog.debug('Kill job %d on pid %d: Could not get lock - injecting backoff messages' % (job.jid, job.pid))                
                        for _ in xrange(self.num_procs*2):
                            self._quick_put(sleep_task_flag) #pump channel with backoff messages
                        timeout = 0.5 #highly likely to succeed now  
                cloudLog.debug('Kill job %d on pid %d: Acquired subprocess input pipe' % (job.jid, job.pid))         
                if job.status in ran_statuses: #verify again
                    return
                
                #Clear out result queue
                job.killMe = False #block re-entry on this job
                self._clear_results()            
                job.killMe = True    
                
                cloudLog.debug('Kill job %d on pid %d: Flushed result queue. jobs is %s' % (job.jid, job.pid, job.status))
                if job.status in ran_statuses: #verify one last time before killing..
                    return                            
                
                #At this point the result queue is clear AND no one can write anymore - kill the process            
                for proc in self._pool:
                    if proc.cloudPid == job.pid:
                        cloudLog.debug('terminating job %d on pid %d', job.jid, job.pid)
                        proc.terminate()
                        self._pool.remove(proc)
                        killed = True                    
                        break
            finally:      
                cloudLog.debug('termination done (killed=%s). releasing locks? %s', killed, not re_enter)          
                if not re_enter:
                    self.inKill = False
                    self._inqueue._rlock.release()
                    if self._outqueue._wlock:
                        self._outqueue._wlock.release()
                    cloudLog.debug('Locks are released!')
            if killed:
                cloudLog.debug('Job was terminated -- respawning, my state is %s. result handler state is %s', self._state, self._task_handler._state)
                job._aborted(status_killed)                
                if self._state or self._task_handler._state:
                    cloudLog.debug('not respawning due to non running state!')
                    return  #get out of here
                self._genprocess()
                cloudLog.debug('process respawned state is %s', self._state)

    def _clear_results(self):
        """Clear results from main queue and process
        Windows note: There appears to be an issue with calling poll() on named pipes in windows
        if another pipe is receiving.  Consequently, we must just sleep and hope that the taskmanager clears        
        """
        get = self._quick_get
        poll = self._outqueue._reader.poll        
        
        thread = threading.current_thread()
        
        if sys.platform != 'win32':
            cloudLog.debug('JobKiller: Clearing queue, entering loop? %s Am I result handler? %s' % (poll(),thread == self._result_handler))
        else: #can't call poll
            cloudLog.debug('JobKiller: Clearing queue,Am I result handler? %s' % (thread == self._result_handler))
        #print 'JobKiller: Clearing queue, entering loop? %s Am I result handler? %s' % (poll(),thread == self._result_handler)
        
        
        if thread != self._result_handler: 
            if sys.platform == 'win32': #we cannot safely call poll on windows
                time.sleep(2.0) #hopefully long enough
                return
            else:
                while poll():  #sleep until result handler gets all jobs
                    time.sleep(0.1)
            return
        
        #otherwise, I am the result handler -- clear the queue
        while poll():            
            try:
                #print 'access task'
                task = get()
                #print 'got task %s' % str(task)
            except (IOError, EOFError):
                debug('clearResults got EOFError/IOError -- exiting')
                return

            if thread._state:
                assert thread._state == pool.TERMINATE
                debug('clearResults on terminating')
                break

            if task is None:
                debug('clearResults got sentinel. reinjecting')
                self._outqueue._writer.send(None) #re-inject sentinel
                break

            job, i, obj = task
            try:
                #print 'cache set'
                self._cache[job]._set(i, obj)
                #print 'end set'
            except KeyError:
                pass   
            
    def _monitor_pool(self):
        """Monitor pool for unexpected terminations."""        
        while True:

            with self._process_monitor_cv:
                self._process_monitor_cv.wait(1.5)
            
            if not pool or self._state in (pool.CLOSE, pool.TERMINATE):
                    #pool may be none during system shutdown
                    break                            
            with self.job_kill_lock: #ensure that no job killing is occuring
                #double check once lock granted (abort if task_handler is terminating)
                if not pool or self._state in (pool.CLOSE, pool.TERMINATE) or self._task_handler._state:
                    #pool may be none during system shutdown
                    break            
                
                for p in self._pool[:]:
                    if not p.is_alive(): #sys.exit event!                        
                        pid = p.cloudPid
                        cloudLog.info('Killed process (%s) detected -- respawning! state is %s', pid, self._state)
                        for _, job in self._cache.items(): #find/delete relevant jobs
                            if job.status == status_processing and job.pid == pid:
                                job._aborted(status_error, CloudException('Job was killed with sys.exit(). Please do not use sys.exit() within cloud called functions'))
                        self._pool.remove(p)
                        self._genprocess() #restart process
    
    @staticmethod
    def _cloud_handle_tasks(taskqueue, task_sem, put, outqueue, pool):
        """Uses semaphore to allow pqueue to run"""
        thread = threading.current_thread()

        for _, taskseq in iter(taskqueue.get, None):
            for _, task in enumerate(taskseq):
                if thread._state:
                    debug('task handler found thread._state != RUN')
                    break
                try:
                    debug('task handler ack sem')
                    task_sem.acquire()
                    debug('task handler got sem')
                    #print 'put', task
                    put(task)
                    debug('task handler put done')
                    #print 'put done'
                except IOError:
                    debug('could not put task on queue')
                    break
            else:
                debug('task handler waiting...')
                continue
            break
        else:
            debug('task handler got sentinel')


        try:
            # tell result handler to finish when cache is empty
            debug('task handler sending sentinel to result handler')
            outqueue.put(None)

            # tell workers there is no more work
            debug('task handler sending sentinel to workers')
            for _ in pool:
                put(None)
        except IOError:
            debug('task handler got IOError when sending sentinels')

        debug('task handler exiting')
        
    @classmethod
    def _terminate_pool(cls, taskqueue, inqueue, outqueue, worker_pool,
                        worker_handler, task_handler, result_handler, cache):
        """This is taken from Python2.7.1 terminate_pool
        Added here as python2.7.2 changes some things
        """
        
        # this is guaranteed to only be called once
        debug('finalizing pool')

        task_handler._state = pool.TERMINATE
        worker_handler._state = pool.TERMINATE
        taskqueue.put(None)                 # sentinel

        debug('helping task handler/workers to finish')
        cls._help_stuff_finish(inqueue, task_handler, len(worker_pool))

        assert result_handler.is_alive() or len(cache) == 0

        result_handler._state = pool.TERMINATE
        outqueue.put(None)                  # sentinel

        if worker_pool and hasattr(worker_pool[0], 'terminate'):
            debug('terminating workers')
            for p in worker_pool:
                p.terminate()

        debug('joining task handler')
        task_handler.join(1e100)

        debug('joining result handler')
        result_handler.join(1e100)

        if worker_pool and hasattr(worker_pool[0], 'terminate'):
            debug('joining pool workers')
            for p in worker_pool:
                p.join()
        debug('everything shut down!')
        

    
    def notify_job_done(self, jid):
        job = self._cache[jid]
    
        #dispatch dependencies here                
        with self.dependency_lock:
            errored = job.status in error_statuses
            depends = self.jobInvDeps.get(job,[])
            for dep in depends:                
                indep = self.jobDeps[dep]
                del indep[indep.index(job)]
                if errored:
                    if dep.status not in ran_statuses:
                        dep._aborted(status_stalled)              
                if not indep: 
                    del self.jobDeps[dep]
                    if dep.status not in ran_statuses:
                        self._queue_job(dep)
                    
            if depends:
                del self.jobInvDeps[job]
                
        status = job.status
        exception = job.result if job.status != status_done else None
        
        loginfo = job.logname, job.logcnt, job.logpid
        
        if job.deleted:
            pass
        elif job.status == status_done:
            self.adapter.cloud.cacheManager.putCached(jid, status=status, exception = exception, 
                                                      runtime=job.runtime, loginfo = loginfo, result = job.result)
        else:
            self.adapter.cloud.cacheManager.putCached(jid, status=status, exception = exception, 
                                                      runtime=job.runtime, loginfo = loginfo)   
            
        if self.join_cv:
            with self.join_cv:
                self.join_cv.notifyAll()         
                                 
        
    def _queue_job(self, job):
        job._status = status_new
        self._taskqueue.put((job.priority, [(job.jid, job.func, job.args, job.kwds, 
                                             job.result_serialization_level, job.logname, job.logpid, job.logcnt)]))

    def cloud_apply_async(self, func, args=(), kwds={}, func_name = None, 
                          label = None, dependencies = [], priority = 5,
                          result_serialization_level = 0, 
                          logname="", logpid = 1, logcnt=0):
        '''
        Asynchronous equivalent of `apply()` builtin
        No callback support!!
        '''
        assert self._state == pool.RUN
        job = CloudJob(self, func, args, kwds, func_name, label, priority,
                       result_serialization_level, logname, logpid, logcnt)
        
        with self.dependency_lock:
            try:
                depJobs = [self.get_job_item(dep) for dep in maybe_xrange_iter(dependencies)]
            except CloudException ,e:
                e.parameter = 'This dependency does not exist.'                 
                raise e       
            runningDeps = []            
            for dep in depJobs:
                status =  dep.status                
                if status in error_statuses:
                    job._aborted(status_stalled)
                    break
                elif status not in ran_statuses:
                    runningDeps.append(dep)            
            if runningDeps:                 
                job._status = status_waiting
                self.jobDeps[job] = runningDeps  #job depends on runningDeps
                for indep in runningDeps:
                    depList = self.jobInvDeps.setdefault(indep, [])
                    depList.append(job)
        if job.status == status_new:
            self._queue_job(job)
        return job

    
    def close(self):
        CloudConnection.close(self)           
        cloudLog.debug('Terminating process pool')
        with self.job_kill_lock:  #avoid race condition with job killing
            self.terminate()
        cloudLog.debug('Notifying process monitor. state now %s', self._state)
        with self._process_monitor_cv:
            self._process_monitor_cv.notify() #force shutdown
        self.cloud_manager.close()
        self._process_monitor.join()    
  
    @staticmethod
    def _getparams(job):
        dependencies = job.get('depends_on',[]) 
        label = job.get('job_label',None)
        priority = job.get('job_priority',5) 
        func_name = job['func_name']
        result_serialization_level = job['fast_serialization']
        return func_name, dependencies, label, priority, result_serialization_level        
        
    
    def job_add(self, params, logdata = None):
        func = params['func']
        args = params['args']
        kwargs = params['kwargs']
        
        func_name, dependencies, label, priority, result_serialization_level = MPConnection._getparams(params)                                       
        
        logprefix, logpid, logcnt = logdata
        
        logname = logprefix + '.%sresult'        
        
        
        #print 'START running add... %s' % params
        #print 'add'
        thejob = self.cloud_apply_async(func=func, args=args, kwds=kwargs, func_name = func_name, 
                                        label = label, dependencies = dependencies, priority = priority,  
                                        result_serialization_level = result_serialization_level,
                                        logname=logname, logpid = logpid, logcnt = logcnt)
        #print 'done'
        #print 'DONE running add... %s' % params
        
        return thejob.jid
            
    def jobs_map(self, params, mapargs, mapkwargs = None, logdata = None):
        """Treat each element of argList as func(*element)
        """
        if not mapargs and not mapkwargs:
            raise ValueError('either mapargs or mapkwargs must be non-None')
        
        func = params['func']

        func_name, dependencies, label, priority, result_serialization_level = MPConnection._getparams(params)
        
        logprefix, logpid, logcnt = logdata
        
        ctr = 0
        outjids = []
        
        if not mapargs:
            mapargs = itertools.repeat(())
        if not mapkwargs:
            mapkwargs = itertools.repeat({})
        
        for args, kwargs in itertools.izip(mapargs, mapkwargs): 
            
            logname = ''.join([logprefix,'.%sjobresult_',str(ctr)])
            thejob = self.cloud_apply_async(func=func, args=args, kwds=kwargs, func_name = func_name, 
                                        label = label, dependencies = dependencies, priority = priority,
                                        result_serialization_level = result_serialization_level, 
                                        logname=logname, logpid = logpid, logcnt=logcnt)
            outjids.append(thejob.jid)
            ctr+=1
        return outjids
            
    def get_job_item(self, jid):
        """Retrieve multiprocessing job from either local or cloud cache"""            
        job = self._cache.get(jid)        
        if job:
            if job.deleted:
                raise CloudException('Does not exist or was purged from cache', jid=jid)
            return job
        #been deleted -- access cache
        job = self.adapter.cloud.cacheManager._getJob(jid)
        if job:
            return job
        raise CloudException('Does not exist or was purged from cache', jid=jid)
        
    def jobs_join(self, jids, timeout=None):
        """cv based joining
        Breaks when all jobs done or immediately if an error occurs or if timeout reached        
        """
        if not self.join_cv:
            return False        
        start_time = time.time()
        while True:
            statuses = []
            hit_error = False
            
            can_timeout = timeout is not None and time.time() - start_time > timeout
            
            for jid in maybe_xrange_iter(jids):
                job = self.get_job_item(jid)                
                if not hit_error and not can_timeout and job.status not in ran_statuses:
                    break                
                if job.status in error_statuses:
                    hit_error = True
                statuses.append(job.status)                
            else: #loop is done
                return statuses

            if can_timeout:
                return statuses
            
            with self.join_cv:
                if timeout is not None:
                    waitTime = max(0.1,timeout - (time.time() - start_time) + 0.1) 
                else:
                    waitTime = None
                self.join_cv.wait(waitTime)                        
    
    def jobs_result(self, jids, by_jid):
        def patchResult(jid):
            job = self.get_job_item(jid)
            
            result = job.result
            if not result:
                raise CloudException('Result was purged from cache', jid=jid)            
            return result
            
        return {'data': [patchResult(jid) for jid in maybe_xrange_iter(jids)],
                'interpretation': [{'datatype': 'python_pickle'} for jid in maybe_xrange_iter(jids)]}
    
    def jobs_kill(self, jids):
        target_jids = maybe_xrange_iter(jids) if jids else list(self._cache.keys())
        return [self._kill_job(jid) for jid in target_jids]        
    
    def jobs_delete(self, jids):
        #Kill anything still running
        for jid in  maybe_xrange_iter(jids):
            job = self.get_job_item(jid)
            if isinstance(job, CloudJob) and job.status not in ran_statuses:
                job.deleted = True
                self._kill_job(jid)
    
    def is_simulated(self):        
        return True 
    
    def needs_restart(self, **kwargs):        
        return False #maybe?    
    
    @staticmethod
    def patch_exception(job):
        """Helper for join/status"""
        if isinstance(job, CloudJob):
            return job.result if job.status == status_error else None
        else:  #from cache
            return job.exception    
    
    def jobs_info(self, jids, info_requested):
        jobs = [self.get_job_item(jid) for jid in maybe_xrange_iter(jids)]
        outdict = {}
        errMsg = 'redirect_job_output must be enabled in cloudconf.py to access stdout'
        for jid, job in itertools.izip(maybe_xrange_iter(jids),jobs):
            dct = {}
            if 'status' in info_requested:
                dct['status'] = job.status
                        
            if 'runtime' in info_requested:
                dct['runtime'] = job.runtime  #runtime is a property in CloudJob -- and therefore dynamically changing
                
            if 'exception' in info_requested:
                dct['exception'] = self.patch_exception(job)
            
            if 'stdout' in info_requested or 'stderr' in info_requested:
                report = self.adapter.report  
                if isinstance(job, CloudJob):
                    logname, logcnt, logpid = job.logname, job.logcnt, job.logpid                    
                else:
                    logname, logcnt, logpid = job.loginfo
                if not report:
                    raise CloudException(errMsg, jid=jid)
                if 'stdout' in info_requested:
                    stdoutFile = report.get_report_file(logname, '.stdout', logcnt, logpid)
                    try:                        
                        dct['stdout'] = tail_file(stdoutFile)
                    except (IOError, OSError):
                        if not isinstance(job, CloudJob): # cached - so job is done
                            raise CloudException('Cloud not load %s. %s' % (stdoutFile, errMsg), jid=jid)                        
                if 'stderr' in info_requested:
                    stderrFile = report.get_report_file(logname, '.stderr', logcnt, logpid)
                    try:                        
                        dct['stderr'] = tail_file(stderrFile)
                    except (IOError, OSError):
                        if not isinstance(job, CloudJob): # cached - so job is done
                            raise CloudException('Cloud not load %s. %s' % (stderrFile, errMsg), jid=jid)
                        
            # n/a results for simulation
            # todo: should be able to access profile
            # todo: consider saving created/endtime/logging
            non_sim = ['created', 'pilog', 'code_version', 'env', 'vol', 'finished', 'logging']
            for key in non_sim:
                if key in info_requested:
                    dct[key] = None 
                                    
            outdict[jid] = dct         
        return outdict                                   
    
    def force_adapter_report(self):
        """
        Should the SerializationReport for the SerializationAdapter be coerced to be instantiated?
        """
        return self.redirect_job_output
        
    def report_name(self):
        return 'Multiprocessing'    

