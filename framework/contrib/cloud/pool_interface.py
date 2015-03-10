"""
A module provided that emulates python multiprocessing's Process Pool interface.
This can be used to trivially replace multiprocessing Pool code.

Note that cloud.setkey must still be used before using this interface.

Not supported (mostly due to not making sense in the PiCloud context).

* Pool initializer
* Passing keyword arguments to function being applied
* callback for map/map_async
* (i)map chunksize -- argument only controls cloud.iresult chunksize
* iresult_undordered -- this is just an ordered iresult
* multiprocessing.TimeoutError -- cloud.CloudTimeoutError is used instead
* join, close and terminate operations

.. warning::
    
    It is highly recommended that you do not use this module and instead use the cloud module directly.
    Much functionality that the cloud interface offers is missing here.
"""
"""
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

from cloud import CloudTimeoutError


from . import _getcloud
import multiprocessing

class AsyncResult(object):
    """Result object that emulates multiprocessing.pool.AsyncResult"""
    
    _jid = None #internal - jid (or jid list) associated with this result
    
    def __init__(self, jid):
        self._jid = jid
    
    def get(self, timeout=None):
        """
        Return result when it arrives.
        If timeout is not None and none arrives, 
        raise multiprocessing.TimeoutError in *timeout* seconds
        """
        return _getcloud().result(self._jid)
        
    def wait(self, timeout=None):
        """
        Wait until result is available or *timeout* seconds pass
        """
        try:
            _getcloud().join(self._jid)
        except CloudTimeoutError:
            pass
        
    def ready(self):
        """Returns true if the job finished (done or errored)"""
        c = _getcloud()
        status = c.status(self._jid)
        if not hasattr(status, '__iter__'):
            return status in c.finished_statuses
        else:
            for s in status:
                if s not in c.finished_statuses:
                    return False
            return True
    
    def successful(self):
        """Returns true if job finished successfully.
        Asserts that job has finished"""
        assert(self.ready())
        status = _getcloud().status(self._jid)
        if not hasattr(status, '__iter__'):
            return status == 'done'
        else:
            for s in status:
                if s != 'done':
                    return False
            return True

    

def apply(func, args=()):
    """
    Equivalent to Multiprocessing apply.
    keyword arguments are not supported        
    """
    
    c = _getcloud()
    jid = c.call(func, *args)
    return c.result(jid)

def apply_async(func, args=(), callback=None):
    """
    Equivalent to Multiprocessing apply_async
    keyword arguments are not supported
    
    callback is a list of functions that should be run on the callee's computer once this job finishes successfully.
        Each callback will be invoked with one argument - the jid of the complete job     
    """

    c = _getcloud()
    jid = c.call(func, _callback = callback, *args)
    return AsyncResult(jid)
    

def map(func, iterable, chunksize=None):
    """
    Equivalent to Multiprocessing map
    chunksize is not used here
    """
    
    c = _getcloud()
    jids = c.map(func, iterable)
    return c.result(jids)

def map_async(func, iterable, chunksize=None):
    """
    Equivalent to Multiprocessing map_async
    chunksize is not used here
    """    
    c = _getcloud()
    jids = c.map(func, iterable)
    return AsyncResult(jids)
    
    
def imap(func, iterable, chunksize = None):
    """
    Equivalent to Multiprocessing imap
    chunksize is used only to control the cloud.iresult stage
    """
    
    c = _getcloud()
    jids = c.map(func, iterable)
    return c.iresult(jids,chunksize)

def imap_unordered(func, iterable, chunksize = None):
    """
    Same as imap
    """
    
    return imap(func, iterable, chunksize)

    


    
    