"""
running_on_cloud version of bucket interface that accesses /bucket mount point to do bucket operations
Do NOT directly use this in your own code!
"""
from __future__ import with_statement
from __future__ import absolute_import

"""
TODO:
Consider having connection straight to the boss to sign requests to get/put
""" 

"""Currently not supported:

-getf with byte ranges
-modifying content-type/encoding (uses s3fs default)
-list/iterlist (fast enough to go to webserver / semantically nontrivial to use s3fs w/ limited keys)
-remove_prefix - s3fs lacks support for batch delete
-make_public & friends - s3fs lacks support
-info - s3fs lacks support for metadata requests
-sync_*: Look into how to do this without bringing the whole file in; see rsync trickery of s3fs
    Right now sync just falls back to always put/get file.
        In general sync is less practical server-side
"""

"""
Copyright (c) 2013 `PiCloud, Inc. <http://www.picloud.com>`_.  All rights reserved.

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


import errno
import os
import time
import logging
from .cloud import CloudException, CloudTimeoutError

cloudLog = logging.getLogger('Cloud.bucket')

# base implementation of bucket 
from . import bucket as base_bucket

# auxilerary functions that don't directly touch website
from .bucket import _compute_md5, _get_effective_obj_path, _putf, _ready_file_path, \
                    S3_URL, TruncatableList, builtin_list

# fallback to default implementations
from .bucket import list, iterlist, remove_prefix, make_public, public_url_folder, \
                    is_public, make_private, info, get_md5, _get_md5

# Error codes used by bucket:
filenametoolong = 491
filenotfound = 492

# human readable messages
hrm = {
       filenametoolong: 'The specified filename was too long.',
       filenotfound : 'The specified object was not found.'
       }


def _emulate_apiexception(error_code):
    error_hrm = hrm[error_code]
    return CloudException(error_hrm, jid=None, status=error_hrm)


bucket_basepath = '/bucket'

def _retry_op(func, excp_class=Exception, _num_retries=2):
    retry_cnt = 0
    while retry_cnt <= _num_retries:
        try:
            retval = func()
        except excp_class, e:
            if retry_cnt == _num_retries:
                raise
            else:
                cloudLog.warning('Received exception %s:%s. Retrying', 
                                 type(e), str(e), exc_info=True)
                time.sleep(0.2)
                continue
        else:
            return retval    

def _get_bucket_path(obj_path, prefix):
    full_obj_path = _get_effective_obj_path(obj_path, prefix)
    return os.path.join(bucket_basepath, full_obj_path)

def putf(f, obj_path, prefix=None, _buffersize = 16384):
    """
    helper for putf.
    Does not support content_type/content_encoding
    """
            
    dest_path = _get_bucket_path(obj_path, prefix)
    
    fsize = 0 # file size. may not be computable 
    
    if isinstance(f, basestring):
        from cStringIO import StringIO        
        f = StringIO(f)
    
    try:
        start_loc = f.tell()
        f.seek(0,2)        
        fsize = f.tell() - start_loc
        f.seek(start_loc)
    except IOError:  
        raise IOError('File object is not seekable. Cannot transmit')

    if fsize > 5000000000:
        raise ValueError('Cannot store bucket objects larger than 5GB on cloud.bucket')
    
    if fsize == 0:
        raise ValueError('Cannot store empty bucket objects')
    
    _ready_file_path(dest_path)
    
    def writer():
        with open(dest_path, 'wb') as destf:
            
            data = f.read(_buffersize)
            while data:         
                destf.write(data)
                data = f.read(_buffersize)
    
    try:
        _retry_op(writer,EnvironmentError)
    finally:
        f.close()
        
def put(file_path, obj_path=None, prefix=None):
    
    if obj_path is None:
        obj_path = os.path.basename(file_path)
    elif not obj_path:
        raise ValueError('Cannot upload bucket object with obj_path "%s"' % obj_path)

    f = open(file_path, 'rb')        
    putf(f, obj_path, prefix)
    
# Serverside md5 computation will take too long to be worthwhile, just always do a put
#  TODO: With boss signing, md5 checks may be doiable
sync_to_cloud = put
    
def exists(obj_path, prefix=None):
    bucket_path = _get_bucket_path(obj_path, prefix)
    return _retry_op(lambda: os.path.exists(bucket_path), EnvironmentError)

def remove(obj_paths, prefix=None):
    """Removes object(s) named ``effective_obj_paths`` from PiCloud bucket
    
    obj_paths can be a single object or a list of objects
    """    
    
    if not hasattr(obj_paths, '__iter__'):
        obj_paths = [obj_paths]
    
    if len(obj_paths) > 10: # if batch, use webserver batch remove
        return base_bucket.remove(obj_paths, prefix)
        
    for obj_path in obj_paths:
        bucket_path = _get_bucket_path(obj_path, prefix)
                
        try:
            os.remove(bucket_path)
        except OSError, oe:
            if oe.errno != errno.ENOENT:
                raise

def get(obj_path, file_path=None, prefix=None, start_byte=0, end_byte=None, _retries=1):
    """Similar to regular get, but bypass md5 checking"""
    
    if start_byte or end_byte:
        # cannot handle ranges with s3fs without pulling entire file in
        # fall back to default implementation        
        return base_bucket.get(obj_path, file_path, prefix, start_byte, end_byte, _retries)
    
    file_path = _ready_file_path(file_path, obj_path)    

    # with s3fs, we are effectively doing a copy
    cloud_file = getf(obj_path, prefix, 0, None)
    
    chunk_size = 16384
    
    f = open(file_path, 'wb')
    
    while True:
        data = cloud_file.read(chunk_size)
        if not data:
            break
        f.write(data)
    
    f.close()
    
# always get on server (md5 check takes too long)
sync_from_cloud = get        

"""get logic"""

# redefined 
class CloudBucketObject(file):
    """A CloudBucketObject provides a file-like interface to the contents of an object
    On the server-side this is just a file subclass made compatible with client-side CloudBucketObject"""
    
    __cached_md5 = None
       
    @property
    def md5(self):
        # interferes with read!
        if self.__cached_md5 == None:
            self.__cached_md5, _, _ = _compute_md5(self)
        return self.__cached_md5

    def filesize(self):
        # not threadsafe with respect to read() 
        start_loc = self.tell()
        self.seek(0,2)
        size = self.tell()
        self.seek(start_loc)
        
        return size
    
    def sizeofchunk(self):
        return self.filesize()
    
    def end(self):
        return self.filesize()

def getf(obj_path, prefix=None, start_byte=0, end_byte=None):
    """
    Retrieve the object referenced by ``effective_obj_path`` from PiCloud.
    Return value is a CloudBucketObject (file-like object) that can be read() to 
    retrieve the object's contents 

    An optional byte_range can be specified using ``start_byte`` and ``end_byte``, 
    where only the data between ``start_byte`` and ``end_byte`` is returned and made 
    accessible to the CloudBucketObject.  The returned CloudBucketObject.tell() will 
    initialized to ``start_byte``.
    
    An ``end_byte`` of None or exceeding file size is interpreted as a request to retrieve to end of file.
    """    
    
    if start_byte or end_byte:
        # cannot handle ranges with s3fs without pulling entire file in
        # fall back to default implementation        
        return base_bucket.getf(obj_path, prefix, start_byte, end_byte)
    
    
    bucket_path = _get_bucket_path(obj_path, prefix)
    
    try:
        return CloudBucketObject(bucket_path, 'rb')
    except OSError, e:
        if e.errno == errno.ENOENT:
            raise _emulate_apiexception(filenotfound)
        else:
            raise

def mpsafe_get(obj_path, file_path=None, prefix=None, start_byte=0, end_byte=None, 
               timeout=None, do_sync=False):
    """Multiprocessing-safe variant of get.
    If do_sync is false:
        Download file if and only if file_path does not exist. (atomically checked)
    If do_sync is true:
        Download file if and only if copy at file_path does not match remote copy (atomically checked)
    Regardless, If another process is downloading, wait until process has finished
    
    If timeout is reached, CloudTimeoutException is raised
    """
    
    # Exact same code as base bucket: Here only so it references different get function
    
    from .util.cloghandler import portalocker
    
    file_path = _ready_file_path(file_path, obj_path)    
    
    # lock the file itself
    try:
        #fd = os.open(file_path, os.O_CREAT|os.O_RDONLY)
        fobj = open(file_path,'a')
    except OSError as oe:
        if oe.errno != errno.EEXIST:
            raise
        
    expire_time = timeout + time.time() if timeout else None 
    while True:
        try:
            portalocker.lock(fobj, (portalocker.LOCK_EX | portalocker.LOCK_NB))
        except portalocker.LockException as le:
            if le.args[0] != portalocker.LockException.LOCK_FAILED:
                raise
            if expire_time and time.time() > expire_time:
                raise CloudTimeoutError('could not acquire file lock within timeout')
            
            time.sleep(0.2) # delay
        else:
            break
            
    # We now have the lock; determine what to do based on file size
    # Zero file size is special as buckets cannot contained zero sized files
    # If file currently is zero-sized, we are the first writer: download it
    #   Else, someone else got to it first    
    fsize = os.path.getsize(file_path)
    if not fsize: # download
        get(obj_path, file_path, prefix, start_byte, end_byte)
    elif do_sync:
        sync_from_cloud(obj_path, file_path, prefix)         
    
    fobj.close()  # releases lock
