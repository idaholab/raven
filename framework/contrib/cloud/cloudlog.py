"""
Cloudlog controls the logging of all cloud related messages

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

import os
import sys
import logging


from . import cloudconfig as cc
from .util import fix_sudo_path

c = \
"""Filename where cloud log messages should be written.
This path is relative to ~/.picloud/"""
logFilename = cc.logging_configurable('log_filename',
                                     default='cloud.log',  #NOTE: will not create directories
                                     comment =c)
c = \
"""Should log_filename (default of cloud.log) be written with cloud log messages?
Note that cloud will always emit logging messages; this option controls if cloud should have its own log."""
saveLog = cc.logging_configurable('save_log',
                                 default=True,
                                 comment=c)
c = \
"""logging level for cloud messages.
This affects both messages saved to the cloud log file and sent through the python logging system.
See http://docs.python.org/library/logging.html for more information"""
logLevel = cc.logging_configurable('log_level',
                                  default=logging.getLevelName(logging.DEBUG),
                                  comment=c)

c = \
"""logging level for printing cloud log messages to console.
Must be equal or higher than log_level"""
printLogLevel = cc.logging_configurable('print_log_level',
                                  default=logging.getLevelName(logging.ERROR),
                                  comment=c)

c = """Number of days to wait until purging old serialization log directories"""
purgeDays = cc.logging_configurable('purge_days',
                                        default=7,
                                        comment =c)

by_pid_dir = 'cloudlog-by-pid'

datefmt = '%a %b %d %H:%M:%S %Y'
logfmt_main = "[%(asctime)s] - [%(levelname)s] - %(name)s: %(message)s"
logfmt_pid = "[%(asctime)s] - [%(levelname)s] - %(name)s(%(process)d): %(message)s"

class NullHandler(logging.Handler):
    """A handler that does nothing"""
    def emit(self, record):
        pass


def purge_old_logs(pid_log_dir, mylog):
    """Remove contents in by-pid purgedays
    Warning: Potential errors if log file has been opened > purgedays and not written to
    Highly unlikely"""
    import errno, stat, time
    
    try:
        logs = os.listdir(pid_log_dir)
    except OSError, e:
        if e.errno != errno.ENOENT:
            mylog.debug('Could not purge %s due to %s', pid_log_dir, str(e))
        return
    
    now = time.time()
    allowed_difference = purgeDays * 24 * 3600 #purge days in seconds

    for log in logs:
        try:
            log_path = os.path.join(pid_log_dir, log)
            stat_result = os.stat(log_path)
        except OSError:
            mylog.warn('Could not stat %s', log, exc_info = True)
            continue        
        if  (now - stat_result.st_ctime) > allowed_difference:
            mylog.debug('Deleting %s (%s days old)', log, (now - stat_result.st_ctime)/(24*3600))
            try:
                os.remove(log_path)
            except OSError:
                mylog.warn('Could not delete %s', log_path, exc_info = True)            
    
    
def init_pid_log(base_path, mylog):
    from .util.cloghandler.cloghandler import ConcurrentRotatingFileHandler
    from warnings import warn
    import errno
        
    my_pid = str(os.getpid())      
    path = os.path.join(base_path, by_pid_dir)  

    try:
        os.makedirs(path)
    except OSError, e: #allowed to exist already
        if e.errno != errno.EEXIST:                
            warn('PiCloud cannot create directory %s. Error is %s' % (path, e))

    
    purge_old_logs(path, mylog)
    
    log_path = os.path.join(path, my_pid + '.log')
    try:
        handler = ConcurrentRotatingFileHandler(log_path, maxBytes=4194304,backupCount=7)
    except Exception, e:
        warn('PiCloud cannot open pid-logfile at %s.  Error is %s' % (log_path, e))
    else:
        #hack for SUDO user
        fix_sudo_path(log_path)
        handler.setFormatter(logging.Formatter(logfmt_main, datefmt =datefmt))
        mylog.addHandler(handler)
    

"""Initialize logging"""
def _init_logging():    
    import errno
    from warnings import warn
    
    mylog = logging.getLogger("Cloud")    
    
    #clear handlers if any exist
    handlers = mylog.handlers[:]
    for handler in handlers:
        mylog.removeHandler(handler)
        handler.close()
        
    if saveLog:              
                                  
        from .util.cloghandler.cloghandler import ConcurrentRotatingFileHandler, concurrent_error
        path = os.path.expanduser(cc.baselocation)
        try:
            os.makedirs(path)
        except OSError, e: #allowed to exist already
            if e.errno != errno.EEXIST:                
                warn('PiCloud cannot create directory %s. Error is %s' % (path, e))
        
        path = os.path.join(path, logFilename)
        
        try:
            handler = ConcurrentRotatingFileHandler(path,maxBytes=4194304,backupCount=7)
        except Exception, e: #warn on any exception            
            warn('PiCloud cannot open logfile at %s.  Error is %s' % (path, e))
            handler = NullHandler()
        else:
            fix_sudo_path(path)
            fix_sudo_path(path.replace('.log', '.lock'))
    else:
        #need a null hander
        handler = NullHandler()
    handler.setFormatter(logging.Formatter(logfmt_pid, datefmt=datefmt))
    mylog.addHandler(handler)
    mylog.setLevel(logging.getLevelName(logLevel))    
    
    #start console logging:
    printhandler = logging.StreamHandler()
    printhandler.setLevel(logging.getLevelName(printLogLevel))
    printhandler.setFormatter(
       logging.Formatter(logfmt_main,
       datefmt= datefmt))
    mylog.addHandler(printhandler)
    
    #If pilog exists, use it as a parent
    try:
        pilog_module = __import__('pimployee.log', fromlist=['log'])
        pilogger = pilog_module.pilogger
        mylog.parent = pilogger
    except (ImportError, AttributeError):
        pass
        
    
        
    if not isinstance(handler, NullHandler):
        mylog.debug("Log file (%s) opened" % handler.baseFilename)
        
        # with pid based logging, this is less relevant now
        if concurrent_error:
            mylog.warning('Could not use ConcurrentRotatingFileHandler due to import error: %s.' +
                          'Likely missing pywin32 packages. Falling back to regular logging; you may experience log corruption if you run multiple python interpreters',
                          concurrent_error)        

    if saveLog:
        init_pid_log(os.path.expanduser(cc.baselocation), mylog)

            
    return mylog
                                                              
cloudLog = _init_logging()


"""verbose mode
Whether the below functions are sent
"""
verbose = cc.logging_configurable('verbose',
                                     default=False, 
                                     comment = "Should cloud library print informative messages to stdout and stderr",
                                     hidden = True)

def stdout(s, auto_newline=True):
    """Write to stdout if verbose
    If auto_newline, add \n to end (like print)"""
    if not verbose:
        return 
    sys.stdout.write(s)
    if auto_newline:
        sys.stdout.write('\n')
    sys.stdout.flush()
               
def stderr(s, auto_newline=True):
    """Write to stderr if verbose
    If auto_newline, add \n to end (like print)"""
    if not verbose:
        return 
    sys.stderr.write(s)
    if auto_newline:
        sys.stderr.write('\n')
    sys.stderr.flush()
