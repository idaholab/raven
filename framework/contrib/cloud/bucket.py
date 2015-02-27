"""
Buckets is a key-value interface to store and retrieve data objects up to 5 GB in size.

A full overview of the bucket interface may be found within PiCloud's 
`documentation <http://docs.picloud.com/bucket.html>`_.

This module provides a python interface to your bucket.

In general, the various functions of this module use one or more of the following parameters:

* ``obj_path``: the key (path) of objects stored within the PiCloud bucket data store
* ``prefix``: Easy way to namespace objects; If present, the ``effective_obj_path`` is prefix + / + obj_path.
    If not present, the ``effective_obj_path`` is just obj_path.  
    Referring to an object by obj_path="file", prefix="folder" is the same as 
    obj_path="folder/file", prefix=""
* ``file_path``: Refers to a path on the local file system.

For convenience, in "copy" type functions (get, put, sync_to_cloud, etc.), the "destination" may be left 
blank. If so, the destination will be set to os.path.basename(source).
(e.g. to upload, you need not put(file_path='foo', obj_path='foo'). put(file_path='foo') will suffice.)  

.. note::
    Objects appearing in the bucket data store do **not** automatically appear on a 
    job's mounted file system. Jobs **must** access bucket data with the get function.    
"""

# TODO: Support files larger than 5 gb

from __future__ import with_statement
from __future__ import absolute_import
"""
Copyright (c) 2012 `PiCloud, Inc. <http://www.picloud.com>`_.  All rights reserved.

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

"""Dev beware: list is defined in this module; list will not map to builtin list. Use builtin_list instead"""

import os
import base64
import sys
import logging
import time
import socket
import errno
import mimetypes
import random
import re
import urllib2
from __builtin__ import list as builtin_list
from ssl import SSLError
from itertools import islice

__httpConnection = None
__url = None

from .transport.adapter import SerializingAdapter
from .transport.network import HttpConnection
from .util import  min_args, max_args
from .util.zip_packer import Packer
from .cloud import CloudException, CloudTimeoutError
from cloud import _getcloudnetconnection, _getcloud

cloudLog = logging.getLogger('Cloud.bucket')

S3_URL = 'https://s3.amazonaws.com/'
_bucket_new_query = 'bucket/new/'
_bucket_list_query = 'bucket/list/'
_bucket_get_query = 'bucket/get/'
_bucket_exists_query = 'bucket/exists/'
_bucket_info_query = 'bucket/info/'
_bucket_md5_query = 'bucket/md5/'
_bucket_remove_query = 'bucket/remove/'
_bucketmap_job_query = 'job/bucketmap/'

_bucket_make_public_query = 'bucket/make_public/'
_bucket_is_public_query = 'bucket/is_public/'
_bucket_make_private_query = 'bucket/make_private/'
_bucket_public_url_folder_query = 'bucket/public_url_folder/'

xml_chars = object # initialized on demand
"""
The functions can be viewed as functionally close to instance methods of cloud.Cloud
"""

def _post(conn, url, post_values, headers={}):
    """Use HttpConnection *conn* to issue a post request at *url* with values *post_values*"""
    
    #remove UNICODE from addresses
    url = url.decode('ascii', 'replace').encode('ascii', 'replace')
    
    if 'success_action_redirect' in headers:
        headers['success_action_redirect'] = headers['success_action_redirect'].decode('ascii', 'replace').encode('ascii', 'replace')
    if post_values and 'success_action_redirect' in post_values:
        post_values['success_action_redirect'] = post_values['success_action_redirect'].decode('ascii', 'replace').encode('ascii', 'replace')
    
    cloudLog.debug('post url %s with post_values=%s. headers=%s' % (url, post_values, headers))
    response =  conn.post(url, post_values, headers, use_gzip=False)
    
    return response

def _aws_retryable_post(conn, url, post_values, headers={}):
    """Wraps _post with ability to retry
    Sets some other necessary AWS settings"""
    retry_attempts = 8 # AWS can have scalability problems
    attempt = 0
    
    # likely not needed?
    #headers["Connection"] = "close"     
    
    while attempt <= retry_attempts:    
        try:
            return _post(conn, url, post_values, headers)
        except Exception, e:
            if isinstance(e, urllib2.HTTPError) and 200 < e.code < 300: # python 2.5 bug
                return e
            
            attempt += 1            
            if attempt > retry_attempts:
                cloudLog.exception('_aws_retryable_post: Cannot connect to AWS')
                raise 
            cloudLog.warn('_aws_retryable_post: Problem connecting to AWS. Retrying. \nError is %s' % str(e))
            c = attempt -1 
            if (isinstance(e, socket.error) and getattr(e, 'errno', e.args[0]) == errno.ECONNREFUSED) or \
                (isinstance(e, urllib2.HTTPError) and e.code in [500, 503]):
                # guarantee at least a 3 second sleep on a connection refused error/500
                time.sleep(min(60, 3 + (1 << c) *random.random()))
            else:
                time.sleep(min(30, (1 << c) *random.random()))
            continue
            
        

class CloudBucketObject(object):
    """A CloudBucketObject provides a file-like interface to the contents of an object
    Seeking is not available."""
    
    __http_response = None

    __file_size = None
    __start_byte = None
    __end_byte = None
    __pos = None # current byte offset into file
    
    def __init__(self, action, ticket, file_size, start_byte=0, end_byte=None):
        """Wraps S3
        Action = url
        ticket = http headers
        """
        if not isinstance(start_byte, (int, long)):
            raise TypeError('start_byte must be an integer')
    
        if end_byte and not isinstance(end_byte, (int, long)):
            raise TypeError('end_byte must be an integer')
        
        self.__action = action
        self.__ticket = ticket
        self.__file_size = file_size
        self.__start_byte = start_byte
        self.__pos = self.__start_byte
        if end_byte:
            self.__end_byte = min(end_byte, file_size)
        else:
            self.__end_byte = file_size
        
        self.__connect()
        
    def __connect(self):
        """Connect to S3"""            
        if self.__http_response:
            self.__http_response.close()
        self.__ticket['Range'] = 'bytes=%s-%s' % tuple(  [self.__pos, self.__end_byte]  )        
        conn = _getcloudnetconnection()
        self.__http_response =  _aws_retryable_post(conn, self.__action, None, self.__ticket)
      
    def __protected_httpresp_op(self,op_name, *args, **kwargs):
        """Call HTTPResponse method name op_name in protected way
        On failure, reconnect
        """
        max_fails = 3
        for fail_num in xrange(max_fails):
            try:
                meth = getattr(self.__http_response, op_name)
#                if random.random() < 0.05:
#                    raise ssl.SSLError('hack')

                return meth(*args, **kwargs)

            except (SSLError, IOError):
                if fail_num == max_fails - 1:
                    raise
                cloudLog.info('Reconnecting to %s after failure %s at position %s',
                              self.__action, fail_num, self.__pos)

                time.sleep((1 << fail_num) * random.random())
                self.__connect()      
                
    def __iter__(self):
        return self

    def close(self):        
        """
        Close the object, blocking further I/O operations
        """
        return self.__http_response.close()
    
    @property
    def md5(self):
        """The md5 checksum of the contents
        Return None if not available"""
        md5str = self.__http_response.headers['etag'].strip('"')
        if '-' in md5str: # multipart; can't use
            return None
        else:
            return md5str 
    
    def next(self):
        data = self.__protected_httpresp_op('next')
        self.__pos += len(data)
        return data
    
    def read(self, size=-1):
        """
        read([size]) -> read at most size bytes, returned as a string.
       
        If the size argument is negative or omitted, read until EOF is reached.
        """
        data = self.__protected_httpresp_op('read', size)
        self.__pos += len(data)
        return data
        
    def readline(self, size=-1):
        """
        readline([size]) -> next line from the file, as a string.
       
        Retain newline.  A non-negative size argument limits the maximum
        number of bytes to return (an incomplete line may be returned then).
        Return an empty string at EOF.
        """
        line = self.__protected_httpresp_op('readline', size)
        self.__pos += len(line)
        return line
        
    def readlines(self, sizehint=0):
        """
        readlines([size]) -> list of strings, each a line from the file.
       
        Call readline() repeatedly and return a list of the lines so read.
        The optional size argument, if given, is an approximate bound on the
        total number of bytes in the lines returned.
        """
        lines = self.__protected_httpresp_op('readlines', sizehint)
        self.__pos += sum((len(line) for line in lines))
        return lines

    def filesize(self):
        return self.__file_size
    
    def sizeofchunk(self):
        return self.__start_byte - self.__end_byte
    
    def tell(self):
        '''Returns current file position as an integer'''
        return self.__pos

    def end(self):
        return self.__end_byte
 
def _compute_md5(f, buffersize = 8192):
    """Computes the md5 hash of file-like object
    f must have seek ability to perform this operation    
    buffersize controls how much of file is read at once
    
    Returns tuple of hexdigest, base64_encoded_md5_hash, size
    """
    
    try:
        start_loc = f.tell()
        f.seek(start_loc)  #test seek
    except (AttributeError, IOError), e:
        raise IOError('%s is not seekable. Cannot compute MD5 hash. Exact error is %s' % (str(f), str(e)))
    
    try:
        from hashlib import md5
    except ImportError:
        from md5 import md5
    
    m = md5()    
    s = f.read(buffersize)
    while s:
        m.update(s)
        s = f.read(buffersize)    
    
    size = f.tell() - start_loc
    f.seek(start_loc)
    hex_md5 = m.hexdigest()    
    base64md5 = base64.encodestring(m.digest())
    return hex_md5, base64md5, size        

def put(file_path, obj_path=None, prefix=None):
    """
    Transfer the file located on the local filesystem at ``file_path`` to the PiCloud bucket.       
    
    The name of the uploaded object will be the ``effective_obj_path`` (see module help)
    If ``obj_path`` is omitted, the name will be the basename of ``file_path``.
    
    The uploaded object can be retrieved later with the get function.
        
    Example::    
    
        cloud.bucket.put('data/names.txt') 
    
    This will upload *data/names.txt* from the local filesystem to the PiCloud bucket 
    and store the object with the effective_obj_path *names.txt*. It can be later retrieved 
    via cloud.bucket.get('names.txt')"""
    
    if obj_path is None:
        obj_path = os.path.basename(file_path)
    elif not obj_path:
        raise ValueError('Cannot upload bucket object with obj_path "%s"' % obj_path)
        
    # open the requested file in binary mode (relevant in windows)
    content_type, content_encoding = mimetypes.guess_type(file_path)
    
    f = open(file_path, 'rb')        
    _putf(f, obj_path, prefix, content_type, content_encoding)

def _validate_xml(s, entity_name):
    """Verify that s can be represented correctly in xml
    All key names must conform to XML 1.0
    """
    global xml_chars
        
    if not isinstance(s, unicode):
        try:
            uni_s = s.decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError('Cannot represent %s in utf-8' % entity_name)
    else:
        uni_s = s
        
        
    if xml_chars == object: # regex not initialized        
        try:
            xml_chars = re.compile(u'[^\x09\x0A\x0D\u0020-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]', re.U)
        except Exception: # narrow unicode python (http://wordaligned.org/articles/narrow-python)
            try: 
                xml_chars = re.compile(u'[^\x09\x0A\x0D\u0020-\uD7FF\uE000-\uFFFD]', re.U)
            except Exception, e: # no idea?
                cloudLog.warning('Could not generate xml regex', exc_info = True)
                xml_chars = None

    elif xml_chars and xml_chars.search(uni_s):
        raise ValueError('%s contains characters illegal in XML 1.0' % entity_name)
    

def _get_effective_obj_path(obj_path, prefix):
    """Convert obj_path, prefix into an effective_obj_path
    Also, sanity check parameters
    """
    
    if not isinstance(obj_path, basestring):
        raise TypeError('obj_path must be a string')    
    if '../..' in obj_path:
        raise ValueError('"../.." cannot be in obj_path')    
    if obj_path.endswith('/'):
        raise ValueError('Cannot end a obj_path with a forward slash')    
    if obj_path.startswith('/'):
        raise ValueError('Cannot begin a obj_path with a forward slash')    
    if '//' in obj_path:
        raise ValueError('Cannot have consecutive forward slashes in obj_path')    
    _validate_xml(obj_path, 'obj_path')
    
    if not prefix:
        return obj_path
    
    if hasattr(prefix, '__iter__'):
        prefix = '/'.join(prefix)
    
    if not isinstance(prefix, basestring):
        raise TypeError('prefix must be a string or list of strings')    
    if '../..' in prefix:
        raise ValueError('"../.." cannot be in prefix')    
    if prefix.startswith('/'):
        raise ValueError('Cannot begin a prefix with a forward slash')
    if '//' in prefix:
        raise ValueError('Cannot have consecutive forward slashes in prefix')
    _validate_xml(prefix, 'prefix')
        
    if not prefix.endswith('/'):
        prefix = prefix + '/'        
    return prefix + obj_path
    

def _putf(f, obj_path, prefix=None, content_type=None, content_encoding=None):
    """
    helper for putf.
    Accepts arbitrary content_type and content_encoding
    """
    
            
    full_obj_path = _get_effective_obj_path(obj_path, prefix)
    
    fsize = 0 # file size. may not be computable 
    
    if isinstance(f, basestring):
        from cStringIO import StringIO        
        f = StringIO(f)
    
    conn = _getcloudnetconnection()         
    
    try:
        #raise IOError
        hex_md5, content_md5, fsize = _compute_md5(f) 
        
    except IOError:  
        raise IOError('File object is not seekable. Cannot transmit')
    
    if fsize > 5000000000:
        raise ValueError('Cannot store bucket objects larger than 5GB on cloud.bucket')
    
    if fsize == 0:
        raise ValueError('Cannot store empty bucket objects')
    
    try:
        cloudLog.debug('bucket object obj_path in client: %s' % full_obj_path)
        # get a file ticket
        resp = conn.send_request(_bucket_new_query, {'name': full_obj_path,
                                                     'content-type' : content_type,
                                                     'content-encoding' : content_encoding,
                                                     'hex-md5' : hex_md5
                                                     })
        ticket = resp['ticket']
        params = resp['params']
        
        url = params['action']
        
        # update ticket
        ticket['file'] = f
        if content_md5:
            ticket['Content-MD5'] = content_md5
                
        resp =  _aws_retryable_post(conn, url, ticket)
        resp.read()
        
    finally:
        f.close()
    

def putf(f, obj_path, prefix=None):
    """
    Similar to put, but accepts any file-like object (file, StringIO, etc.) ``f`` 
    in lieu of a file_path.  The contents of the uploaded object will be
    f.read()

    The file-like object ``f`` must be seekable. Note that after putf returns, f
    will be closed
    """
        
    # infer mime types
    content_type, content_encoding = mimetypes.guess_type(obj_path)

    return _putf(f, obj_path, prefix, content_type, content_encoding)

def sync_to_cloud(file_path, obj_path=None, prefix=None):
    """
    Update bucket object if it has changed.
    
    put(file_path, obj_path, prefix) only if contents of ``file_path`` on local file system 
    differ from bucket object ``effective_obj_path`` (or bucket object does not exist)
    """
    
    if not obj_path:
        obj_path = os.path.basename(file_path)   
    
    # open the requested file in binary mode (relevant in windows)
    f = open(file_path, 'rb')
    local_md5, _, _ = _compute_md5(f)
    try:
        remote_md5 = _get_md5(obj_path, prefix, log_missing_file_error = False)
    except CloudException: #file not found
        remote_md5 = ''
    do_update = remote_md5 != local_md5
    
    cloudLog.debug('remote_md5=%s. local_md5=%s. uploading? %s',
                   remote_md5, local_md5, do_update
                   )
    if do_update:
        putf(f, obj_path, prefix)
 
class TruncatableList(builtin_list):
    truncated = False 
 
def list(prefix=None, folderize=False, marker=None, max_keys=1000):
    """
    Retrieve obj_paths of all objects stored on PiCloud. Returns a list of keys in 
    lexicographic order. 
        
    * ``prefix``: Return only keys beginning with prefix. 
    * ``folderize``: Treat listing as directory based; compact keys containing "/" into a single folder
        (a key is a folder if and only if it ends in "/")  
        A folder can then be inspected by setting prefix equal to the folder name
    * marker: Return only keys with where key > marker
    * max_keys: Maximum number of keys that can be returned (max 1000). Fewer may be returned
     
    The list will have an attribute, *truncated*, that indicates if the listing is truncated.
    To see the next results, make a subsequent list query with marker set to list[-1]
    
    Use *iterlist* to avoid truncation 
    """
    
    conn = _getcloudnetconnection()
    
    if max_keys > 1000:
        max_keys = 1000

    resp = conn.send_request(_bucket_list_query, {'prefix': prefix,
                                                  'delimiter': '/' if folderize else None,
                                                  'marker': marker,
                                                  'max_keys': max_keys})
    
    files = TruncatableList(resp['files'])
    truncated = resp['truncated']

    files.truncated = truncated
    return files

def iterlist(prefix=None, folderize=False, num_readahead_keys=1000):
    """Retrieve obj_paths of all objects stored on PiCloud. Returns an iterator that produces keys 
    in lexicographic order.  Unlike list() this guarantees that all keys will be returned:
    
    * ``prefix``: Return only keys beginning with prefix. 
    * ``folderize``: Treat listing as directory based; compact keys containing "/" into a single folder
        (a key is a folder if and only if it ends in "/")  
        A folder can then be inspected by setting prefix equal to the folder name
    * ``num_readahead_keys``: Number of keys to prefetch from amazon
    
    .. warning:: Usage of iterlist is not recommended if result set will exceed 3,000 entries.
        Consider using prefixes and folderize to narrow results
    """
    
    def listgen():
        marker = None
        truncated = True
        while truncated:
            obj_name_chunk = list(prefix, folderize, marker, num_readahead_keys)
            truncated = obj_name_chunk.truncated
            for obj_name in obj_name_chunk:
                yield obj_name
            if not truncated:
                break
            marker = obj_name            
            
    return listgen()

def exists(obj_path, prefix=None):
    """Return boolean indicating if PiCloud bucket object named ``effective_obj_path`` exists"""
    conn = _getcloudnetconnection()
        
    full_obj_path = _get_effective_obj_path(obj_path, prefix)
    resp = conn.send_request(_bucket_exists_query, {'name': full_obj_path})
    exists = resp['exists']
    return exists

def info(obj_path, prefix=None):
    """Return information about the PiCloud bucket object ``effective_obj_path``
    
    Information includes size, created time, last modified time, md5sum, public URL (if any), 
    and any headers set with ``make_public``      
    """
     
    conn = _getcloudnetconnection()
        
    full_obj_path = _get_effective_obj_path(obj_path, prefix)
    resp = conn.send_request(_bucket_info_query, {'name': full_obj_path})
    del resp['data']
    if 'url' in resp:
        resp['url'] = S3_URL+resp['url']
    
    return resp

def _get_md5(obj_path, prefix=None, log_missing_file_error = True):
    conn = _getcloudnetconnection()
    
    full_obj_path = _get_effective_obj_path(obj_path, prefix)
    resp = conn.send_request(_bucket_md5_query, {'name': full_obj_path},
                             log_cloud_excp = log_missing_file_error)
    md5sum = resp['md5sum']
    
    if '-' in md5sum: # multipart; can't rely on md5
        return None
    
    return md5sum

def get_md5(obj_path, prefix=None):
    """Return the md5 checksum of the PiCloud bucket object ``effective_obj_path``
    Return None if not available
    """
    return _get_md5(obj_path, prefix)
    
def remove(obj_paths, prefix=None):
    """Removes object(s) named ``effective_obj_paths`` from PiCloud bucket
    
    obj_paths can be a single object or a list of objects
    """
    conn = _getcloudnetconnection()
    
    if not hasattr(obj_paths, '__iter__'):
        obj_paths = [obj_paths]
        
    obj_paths_iter = obj_paths.__iter__()
    
    removed = False
    while True:
        paths_to_remove = builtin_list(islice(obj_paths_iter, 1000))
        if not paths_to_remove:
            break

        full_obj_paths =[ _get_effective_obj_path(obj_path, prefix) for obj_path in paths_to_remove]
        resp = conn.send_request(_bucket_remove_query, {'name': full_obj_paths})
        removed = resp['removed']
    
    return removed

def remove_prefix(prefix):
    """Removes all objects beginning with prefix"""
    objects_iter = iterlist(prefix)
    
    while True:
        batched_objects = builtin_list(islice(objects_iter, 1000))
        if not batched_objects:
            break
    
        remove(batched_objects)

def make_public(obj_path, prefix=None, headers={}, reset_headers = False):
    """Makes the PiCloud bucket object ``effective_obj_path`` publicly accessible by a URL
    Returns public URL
    
    Additionally, you can control the HTTP headers that will be in the response to a request 
    for the URL with the ``headers`` dictionary.
    
    Possible standard HTTP headers are:
    
    * content-type
    * content-encoding  
    * content-disposition
    * cache-control    
    All other headers are considered custom and will have x-amz-meta- prepended to them.
    
    Example:
    make_public('foo',headers={'content-type': 'text/x-python', 'purpose' : 'basic_script'}
    might return \https://s3.amazonaws.com/pi-user-buckets/ddasy/foo
    
    The headers in the response to a request for \https://s3.amazonaws.com/pi-user-buckets/ddasy/foo 
    will include: 
    
    * content-type: text/x-python
    * x-amz-meta-purpose: basic_script
    
    Clear all custom headers, other than content-type and content-encoding, by 
    setting ``reset_headers`` to True 
    
    .. note:: Default content-type and content-encoding are inferred during the original 
        cloud.bucket.put(..) call from the ``file_path`` and ``obj_path``.     
    """
    conn = _getcloudnetconnection()
    
    full_obj_path = _get_effective_obj_path(obj_path, prefix)
    post_values = {'name' : full_obj_path,
                   'reset_headers' : reset_headers}
    for key, val in headers.items():
        try:
            post_values['bh_' + key] = val.decode('ascii').encode('ascii')
        except (UnicodeDecodeError, UnicodeEncodeError):
            raise TypeError('header values must be ASCII strings')                    
    
    resp = conn.send_request(_bucket_make_public_query, post_values)
    public_url = S3_URL+resp['url']
    return public_url

def public_url_folder():
    """Return HTTP path that begins all your public bucket URLs.
    e.g. object 'foo' (if is_public) will be found at
        public_url_folder() + foo
    """
    conn = _getcloudnetconnection()
    resp = conn.send_request(_bucket_public_url_folder_query, {})
    return S3_URL+resp['url']
    
def is_public(obj_path, prefix=None):
    """Determine if the PiCloud bucket object ``effective_obj_path`` 
    is publicly accessible by a URL
    
    Return public URL if it is; otherwise False    
    """
    conn = _getcloudnetconnection()
    
    full_obj_path = _get_effective_obj_path(obj_path, prefix)
    resp = conn.send_request(_bucket_is_public_query, {'name': full_obj_path})
    if resp['status']:
        public_url = S3_URL+resp['url']
        return public_url
    else:
        return resp['status']
    
    
def make_private(obj_path, prefix=None):
    """Removes the public URL associated with the PiCloud bucket object ``effective_obj_path``"""
    conn = _getcloudnetconnection()
    
    full_obj_path = _get_effective_obj_path(obj_path, prefix)
    resp = conn.send_request(_bucket_make_private_query, {'name': full_obj_path})

def _ready_file_path(file_path, obj_path=None):
    """Ensure that the directory where file_path is pointed to exists (create it it doesn't)
    Return file_path (extract it from obj_path if it doesn't exist) 
    """
    if not file_path:
        file_path = os.path.basename(obj_path)
            
    # create directory if needed
    dest_dir = os.path.dirname(file_path)
    if dest_dir:
        try:
            os.makedirs(dest_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
    
    return file_path
    
    
def get(obj_path, file_path=None, prefix=None, start_byte=0, end_byte=None, _retries=1):
    """
    Retrieve the bucket object referenced by ``effective_obj_path`` from PiCloud and 
    save it to the local file system at ``file_path``.
        
    Example::    
    
        cloud.bucket.get('names.txt','data/names.txt') 
    
    This will retrieve the *names.txt* bucket object from PiCloud and save it locally to 
    *data/names.txt*. 

    If ``file_path`` is None, it will be set to the basename of ``obj_path``.
    
    An optional byte_range can be specified using ``start_byte`` and ``end_byte``,
    where only the data between ``start_byte`` and ``end_byte`` is returned. 
    
    An ``end_byte`` of None or exceeding file size is interpreted as a request to retrieve to end of file.
    """
    
    file_path = _ready_file_path(file_path, obj_path)    

    cloud_file = getf(obj_path, prefix, start_byte, end_byte)
    remote_md5 = cloud_file.md5
    
    chunk_size = 64000
    
    f = open(file_path, 'wb')
    
    while True:
        data = cloud_file.read(chunk_size)
        if not data:
            break
        f.write(data)
    
    f.close()
    
    # Only possible to validate MD5 for full file
    if remote_md5 and not start_byte and not end_byte:        
        f = open(file_path, 'rb')
        hex_md5, _, _ = _compute_md5(f)
        f.close()
        
        if hex_md5 != remote_md5:
            msg = 'Local MD5 %s did not match remote MD5 %s for %s' % (hex_md5, remote_md5, file_path)
            cloudLog.error(msg)
            if _retries > 0:
                return get(obj_path, file_path, prefix, start_byte, end_byte, _retries - 1)
                
            raise RuntimeError(msg)
    
    
def sync_from_cloud(obj_path, file_path=None, prefix=None):
    """ 
    Download bucket object if it has changed.
    
    get(obj_path, file_path, prefix) only if contents of ``file_path`` on local file system 
    differ from bucket object ``effective_obj_path`` (or local file does not exist)
    """
    
    file_path = _ready_file_path(file_path, obj_path)
    
    if not os.path.exists(file_path):
        do_update = True
    else:
        f = open(file_path, 'rb')
        local_md5, _, _ = _compute_md5(f)
        f.close()
        try:
            remote_md5 = _get_md5(obj_path, prefix, log_missing_file_error=False)
        except CloudException: #file not found
            remote_md5 = ''
    
        do_update = remote_md5 != local_md5
        cloudLog.debug('remote_md5=%s. local_md5=%s. downloading? %s',
                       remote_md5, local_md5, do_update)
    if do_update:
        get(obj_path, file_path, prefix)
        
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
    
    full_obj_path = _get_effective_obj_path(obj_path, prefix)
    conn = _getcloudnetconnection()

    resp = conn.send_request(_bucket_get_query, {'name': full_obj_path})
    
    ticket = resp['ticket']
    params = resp['params']
    file_size = params['size']
    
    if not start_byte:
        start_byte = 0
        
    if file_size and (not end_byte or end_byte > file_size):
        end_byte = file_size

    cloud_file = CloudBucketObject( params['action'], ticket, file_size, start_byte, end_byte )
    
    return cloud_file

